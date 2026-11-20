from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class TopicRelatednessScorer:
    """文埋め込みのコサイン類似度を返す評価器。"""

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-mpnet-base-v2"
        ),
        device: str | None = None,
        revision: str | None = None,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=device,
            revision=revision,
        )

    def _validate_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("入力は文字列で指定してください。")

        if not text.strip():
            raise ValueError("空の文章は評価できません。")

        max_length = self.model.max_seq_length
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("モデルの最大入力長を確認できません。")
        token_ids = self.model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        if len(token_ids) > max_length:
            raise ValueError(
                f"入力がモデルの上限を超えています: "
                f"{len(token_ids)} > {max_length} tokens。"
                "分割方法を決めてから評価してください。"
            )

    def score(self, C: str, P: str) -> float:
        """文章Cと文章Pの類似度を返す。"""
        return self.score_pairs([(C, P)])[0]

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 32,
    ) -> list[float]:
        """各(C, P)ペアの類似度を入力順に返す。Cは注入前の本文。"""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_sizeは1以上の整数で指定してください。")
        pairs = list(pairs)

        if not pairs:
            return []

        texts: list[str] = []
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("各入力は本文CとプローブPのペアで指定してください。")
            for text in pair:
                if not isinstance(text, str):
                    raise TypeError("入力は文字列で指定してください。")
                if not text.strip():
                    raise ValueError("空の文章は評価できません。")
                texts.append(text)
        unique_texts = list(dict.fromkeys(texts))

        for text in unique_texts:
            self._validate_text(text)

        embeddings = self.model.encode(
            unique_texts,
            batch_size=batch_size,
            prompt="",
            processing_kwargs={"text": {"truncation": False}},
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings)
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(unique_texts)
            or embeddings.shape[1] == 0
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("モデルが不正な埋め込みを返しました。")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("コサイン類似度を計算できない埋め込みです。")
        vectors = dict(zip(unique_texts, embeddings / norms[:, None], strict=True))

        return [
            float(np.clip(np.dot(vectors[C], vectors[P]), -1.0, 1.0))
            for C, P in pairs
        ]