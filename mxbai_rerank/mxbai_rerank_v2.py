from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import set_verbosity_error

from mxbai_rerank.base import BaseReranker
from mxbai_rerank.utils import TorchModule, auto_device, ensure_multiple_of_8, sigmoid_normalize

try:
    from flash_attn import __version__  # noqa: F401

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


estimated_max_cfg = {
    "mixedbread-ai/mxbai-rerank-base-v2": 9.0,
    "mixedbread-ai/mxbai-rerank-large-v2": 12.0,
}


@dataclass
class RerankerOutput:
    """Output dataclass for the classifier model."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None


class MxbaiRerankV2(BaseReranker, TorchModule):
    sep = "\n"
    instruction_prompt = "instruction: {instruction}"
    query_prompt = "query: {query}"
    doc_prompt = "document: {document}"
    task_prompt = "You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:"  # noqa: E501
    chat_template: ClassVar[Dict[str, str]] = {
        "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",  # noqa: E501
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    }

    def __init__(
        self,
        model_name_or_path: str = "mixedbread-ai/mxbai-rerank-base-v2",
        *,
        device: str = auto_device(),
        torch_dtype: str | torch.dtype = "auto",
        max_length: int = 8192,
        tokenizer_kwargs: Optional[dict] = None,
        disable_transformers_warnings: bool = False,
        estimated_max: float | None = None,
        **kwargs,
    ):
        """Initialize the classifier model.

        Args:
            model_name_or_path: Base model to use
            device: Device to use
            max_length: Maximum sequence length
            tokenizer_kwargs: Additional keyword arguments for the tokenizer
            **kwargs: Additional keyword arguments for the model
        """
        TorchModule.__init__(self)
        tokenizer_kwargs = tokenizer_kwargs or {}
        estimated_max = estimated_max or estimated_max_cfg.get(model_name_or_path, 12.0)

        if disable_transformers_warnings:
            set_verbosity_error()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **{
                "attn_implementation": "flash_attention_2" if FLASH_ATTN_AVAILABLE else None,
                "torch_dtype": torch_dtype,
                "device_map": device,
                **kwargs,
            },
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", **tokenizer_kwargs)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True  # 消除分词器fast警告
        self.cfg = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        self.max_length = max_length or self.cfg.max_position_embeddings  # 传入max_length时，取传入值；否则取模型最大输入长度
        self.model_max_length = self.cfg.max_position_embeddings  # 模型最大输入长度
        self.estimated_max = estimated_max

        self.prepare_predefined_inputs()
        self.to(self.model.device, dtype=self.model.dtype)
        self.eval()

    def prepare_predefined_inputs(self):
        """Pre-tokenize static prompts and templates for efficiency."""

        def get_input_ids(x):
            return self.tokenizer(x, return_tensors=None, add_special_tokens=False)["input_ids"]

        self.yes_loc = get_input_ids("1")[0]
        self.no_loc = get_input_ids("0")[0]

        self.task_prompt_inputs = get_input_ids(self.task_prompt)
        self.sep_inputs = get_input_ids(self.sep)
        self.chat_template_prefix_inputs = get_input_ids(self.chat_template["prefix"])
        self.chat_template_suffix_inputs = get_input_ids(self.chat_template["suffix"])

        # Calculate total length of static tokens
        self.predefined_length = (
            len(self.chat_template_prefix_inputs)
            + len(self.task_prompt_inputs)
            + len(self.chat_template_suffix_inputs)
            + len(self.sep_inputs)
        )  # 计算固定token的长度

        # Ensure that the template will not cause the input to exceed the model max length
        # max_length与固定token的长度之和超过模型最大输入长度时，将max_length缩减到恰当值
        if self.max_length + self.predefined_length > self.model_max_length:
            self.max_length = self.model_max_length - self.predefined_length

        self.max_length_padding = ensure_multiple_of_8(
            max(self.model_max_length, self.max_length + self.predefined_length),
            max_value=self.model_max_length,
        )  # 确保填充的最大长度为8的整数倍

    def concat_input_ids(self, input_ids: List[int]) -> List[int]:
        """Concatenate input IDs with prompt templates."""
        return (
            self.chat_template_prefix_inputs
            + input_ids
            + self.sep_inputs
            + self.task_prompt_inputs
            + self.chat_template_suffix_inputs
        )

    def prepare_inputs(self, queries: List[str], documents: List[str], *, instruction: Optional[str] = None) -> dict:  # 对查询列表和文档列表两两配对并进行分词编码
        """Prepare model inputs from query-document pairs.

        Args:
            queries: List of queries
            documents: List of documents
            instruction: Optional instruction

        Returns:
            dict: Tokenized and padded inputs
        """
        inputs = []
        instruction_prompt = self.instruction_prompt.format(instruction=instruction) if instruction else None  # 依据入参设置说明提示词

        for query, document in zip(queries, documents):  # 两两配对进行分词编码
            query_prompt = self.query_prompt.format(query=query)  # 构造查询提示词
            if instruction_prompt:  # 说明提示词非空时，以换行符将其与查询提示词前后拼接
                query_prompt = "".join([instruction_prompt, self.sep, query_prompt])

            # Tokenize query with length limit
            query_inputs = self.tokenizer(
                query_prompt,
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.max_length * 3 // 4,  # Reserve more space for document
                truncation=True,
            )  # 对查询提示词进行分词编码

            available_tokens = self.model_max_length - len(query_inputs["input_ids"]) - self.predefined_length
            doc_maxlen = min(available_tokens, self.max_length)  # 基于长度限制进行最大长度确认

            # Tokenize document
            document_inputs = self.tokenizer(
                self.doc_prompt.format(document=document),
                return_tensors=None,
                add_special_tokens=False,
                max_length=doc_maxlen,  # Avoid exceeding the model maximum length
                truncation=True,
            )  # 对文档提示词进行分词编码

            # Combine query and document
            item = self.tokenizer.prepare_for_model(
                query_inputs["input_ids"],
                self.sep_inputs + document_inputs["input_ids"],
                truncation="only_second",
                max_length=self.max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )  # 将查询提示词、换行符、文档提示词的分词编码结果进行整合

            # Add prompt templates
            # 构造当前查询-文档对的分词编码并收集
            item["input_ids"] = self.concat_input_ids(item["input_ids"])
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)

        # Pad all sequences to same length
        return self.tokenizer.pad(
            inputs,
            padding="longest",
            max_length=self.max_length_padding,
            pad_to_multiple_of=8,  # For efficient tensor operations
            return_tensors="pt",
        )  # 填充以使得所有输入的分词编码在同一长度

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入编码
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> RerankerOutput:
        """Forward pass of the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for training

        Returns:
            RerankerOutput containing loss, logits and predictions
        """
        outputs = self.model(
            input_ids=input_ids,  # 维度：batch_size, sequence_length
            attention_mask=attention_mask,
        )  # 模型推理，输出outputs.logits维度：batch_size, sequence_length, vocab_size

        # Use logits for classification
        # 计算相似度评分，outputs.logits是未经过归一化的预测值，`outputs.logits`的形状通常为`[batch_size, sequence_length, vocab_size]`，即每个token位置上对应词表大小的logits。
        yes_logits = outputs.logits[:, -1, self.yes_loc]  # 这里只关注序列最后一个token（`[:, -1, :]`）在特定位置`self.yes_loc`的logits，也即`yes_logits`：获取每个样本中最后一个token对应“是”的logit值。
        no_logits = outputs.logits[:, -1, self.no_loc]  # 这里只关注序列最后一个token（`[:, -1, :]`）在特定位置`self.no_loc`的logits，也即`no_logits`：获取每个样本中最后一个token对应“否”的logit值。
        """
        - 计算每个样本的“是”logit减去“否”logit的差值。这个差值可以理解为模型对“是”的倾向程度：

        - 如果差值为正，表示模型更倾向于“是”；

        - 如果差值为负，表示模型更倾向于“否”。
        """
        logits = yes_logits - no_logits

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.float().view(-1))

        # Get binary predictions
        # 首先对模型预测值进行sigmoid函数处理，将其映射到(0,1)区间，然后基于是否大于0.5做出二分类判断
        predictions = (torch.sigmoid(logits) > 0.5).long()

        return RerankerOutput(loss=loss, logits=logits, predictions=predictions)

    @torch.inference_mode()
    def predict(
        self,
        queries: list[str],
        documents: list[str],
        *,
        instruction: Optional[str] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Get model predictions for query-document pairs."""
        # 查询列表和文档列表一一对应配对，然后分词编码【将查询和文档分别置入相应提示词中，然后分别分词编码，接着进行合并填充】
        inputs = self.prepare_inputs(queries=queries, documents=documents, instruction=instruction)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 注意这里的inputs维度：batch_size*单个序列的分词编码长度
        # 将查询-文档对的分词编码作为输入，推理得到相似度评分
        scores = self.forward(**inputs).logits.cpu().float()
        if normalize:
            scores = self.normalize_scores(scores)
        return scores

    def normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores using sigmoid normalization.

        Args:
            scores: Scores to normalize

        Returns:
            Normalized scores
        """
        return sigmoid_normalize(scores, estimated_max=self.estimated_max)
