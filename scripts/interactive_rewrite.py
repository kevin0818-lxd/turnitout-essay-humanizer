#!/usr/bin/env python3
"""
Turnitout Essay Humanizer — interactive sentence-level paraphrase via MLX.

Each sentence is rewritten into 3 diverse options; user picks the best one.
Ships with a fine-tuned LoRA adapter for Gemma-2-9B-IT (4-bit).

Usage:
  pip install -r requirements.txt
  python scripts/interactive_rewrite.py --input essay.txt --output rewritten.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
except ImportError:
    mlx_load = None
    mlx_generate = None

# --- citation / thinking helpers -------------------------------------------------

_CITATION_RE = re.compile(
    r"\([A-Za-z][^)\n]{0,200}?\d{4}[a-z]?[^)\n]{0,40}?\)"
)


def sentence_has_citation(s: str) -> bool:
    return bool(_CITATION_RE.search(s))


def strip_thinking_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<\|think\|>.*?<\|/think\|>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<\|think\|>.*$", "", cleaned, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


# Conservative academic sentence split (handles common abbrevs poorly but safe default)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'0-9])")


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_sentences(paragraph: str) -> List[str]:
    sents = _SENT_SPLIT_RE.split(paragraph.strip())
    return [s.strip() for s in sents if s.strip()]


def _key(para_idx: int, sent_idx: int) -> str:
    return f"p{para_idx}s{sent_idx}"


def _mlx_gen(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float = 0.9,
) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = (
            "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
            + "\n\nASSISTANT:\n"
        )

    kwargs = dict(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        verbose=False,
    )
    try:
        return str(
            mlx_generate(model, tokenizer, prompt=prompt, **kwargs)
        ).strip()
    except TypeError:
        try:
            return str(mlx_generate(model, tokenizer, prompt, **kwargs)).strip()
        except TypeError:
            return str(
                mlx_generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
            ).strip()


SYS_SINGLE_PARAPHRASE = """You are an academic writing assistant. Rewrite the given English sentence into natural academic English.
- Keep the same meaning and claims.
- Do not add facts, citations, numbers, or examples.
- Preserve proper nouns and technical terms.
- Output only one rewritten English sentence.
- Do not use bullets, numbering, labels, JSON, or explanation."""

SYS_CN2EN = """You translate Chinese academic text into English. Preserve proper nouns, citation parentheticals, technical terms, and numbers exactly as implied by the source. Output only the English translation with no preamble, quotes, or explanation."""


_CN_TRANSLATE_PROMPTS = [
    "Translate the following English sentence into natural simplified Chinese for an academic context. Output ONLY the Chinese sentence, no quotes or explanation.\n\n{sentence}",
    "请将下面的英文学术句子翻译为简体中文，只输出中文译文，不加引号或解释。\n\n{sentence}",
    "把以下英文句子用不同的中文表达方式翻译出来，保持学术语体，仅输出中文。\n\n{sentence}",
]


def _clean_candidate(raw: str) -> str:
    text = strip_thinking_tags(raw).strip()
    text = re.sub(r"^```(?:text|json)?", "", text, flags=re.I).strip()
    text = text.strip("`").strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    text = text.splitlines()[0].strip() if text.splitlines() else text
    text = re.sub(
        r"^(?:option|candidate|paraphrase|rewrite(?:n)?|response)\s*\d*\s*[:\-]\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^(?:[ABC]|\d+)[\.\:\-\)]\s*", "", text, flags=re.I)
    return text.strip(" \"'")


def _normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]+", "", text)
    return text.strip(" .,!?:;")


def _translate_en_to_cn(
    model, tokenizer, sentence: str, *, prompt_template: str,
    max_tokens: int, temperature: float, top_p: float = 0.92,
) -> str:
    messages = [
        {"role": "user", "content": prompt_template.format(sentence=sentence.strip())},
    ]
    raw = _mlx_gen(model, tokenizer, messages,
                   max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    line = strip_thinking_tags(raw).split("\n")[0].strip()
    return line


def generate_three_options(
    model,
    tokenizer,
    sentence: str,
    *,
    max_tokens: int,
    base_temp: float,
    generation_mode: str = "direct",
    retries: int = 2,
) -> Tuple[str, str, str]:

    prompt_variants = [
        "Sentence:\n{sentence}",
        "Rewrite this sentence into natural academic English. Output only the rewritten sentence.\n\n{sentence}",
        "Provide one distinct academic paraphrase of the sentence below. Keep the meaning. Output only one sentence.\n\n{sentence}",
    ]
    temperatures = [base_temp, min(base_temp + 0.15, 0.95), min(base_temp + 0.30, 0.98)]

    options: List[str] = []
    seen = {_normalize(sentence)}

    if generation_mode == "cn2en-bridge":
        cn_temps = [0.3, 0.55, 0.80]
        cn_top_ps = [0.85, 0.92, 0.97]
        cn_versions: List[str] = []
        cn_seen: Set[str] = set()
        for i in range(3):
            cn = _translate_en_to_cn(
                model, tokenizer, sentence,
                prompt_template=_CN_TRANSLATE_PROMPTS[i],
                max_tokens=min(max_tokens, 256),
                temperature=cn_temps[i],
                top_p=cn_top_ps[i],
            )
            if cn and cn not in cn_seen:
                cn_versions.append(cn)
                cn_seen.add(cn)
            else:
                cn_retry = _translate_en_to_cn(
                    model, tokenizer, sentence,
                    prompt_template=_CN_TRANSLATE_PROMPTS[0],
                    max_tokens=min(max_tokens, 256),
                    temperature=min(cn_temps[i] + 0.2, 0.95),
                    top_p=0.95,
                )
                cn_versions.append(cn_retry if cn_retry else cn or "")

        for idx, cn_text in enumerate(cn_versions):
            if not cn_text:
                options.append(sentence.strip())
                continue
            candidate = ""
            for attempt in range(retries + 1):
                temp = min(temperatures[idx] + 0.06 * attempt, 0.98)
                messages = [
                    {"role": "user", "content": f"{SYS_CN2EN}\n\n{cn_text}"},
                ]
                raw = _mlx_gen(model, tokenizer, messages,
                               max_tokens=max_tokens, temperature=temp,
                               top_p=min(0.88 + 0.04 * idx, 0.99))
                cleaned = _clean_candidate(raw)
                normal = _normalize(cleaned)
                if len(cleaned.split()) < 3 or not normal or normal in seen:
                    continue
                candidate = cleaned
                seen.add(normal)
                break
            options.append(candidate or sentence.strip())

    else:
        for idx in range(3):
            prompt = prompt_variants[idx].format(sentence=sentence.strip())
            candidate = ""
            for attempt in range(retries + 1):
                temp = min(temperatures[idx] + 0.08 * attempt, 0.98)
                messages = [
                    {"role": "user",
                     "content": f"{SYS_SINGLE_PARAPHRASE}\n\n{prompt}"},
                ]
                raw = _mlx_gen(model, tokenizer, messages,
                               max_tokens=max_tokens, temperature=temp,
                               top_p=min(0.9 + 0.03 * idx, 0.99))
                cleaned = _clean_candidate(raw)
                normal = _normalize(cleaned)
                if len(cleaned.split()) < 3 or not normal or normal in seen:
                    continue
                candidate = cleaned
                seen.add(normal)
                break
            options.append(candidate or sentence.strip())

    while len(options) < 3:
        options.append(sentence.strip())
    return (options[0], options[1], options[2])


def translate_en_to_cn_placeholder(
    model, tokenizer, sentence: str, *,
    max_tokens: int, temperature: float,
) -> str:
    return _translate_en_to_cn(
        model, tokenizer, sentence,
        prompt_template=_CN_TRANSLATE_PROMPTS[0],
        max_tokens=max_tokens, temperature=temperature,
    )


def parse_boundary_paras(spec: str, n_paras: int) -> Set[int]:
    """Return 0-based paragraph indices to apply Chinese boundary."""
    spec = spec.strip().lower()
    if not spec:
        return set()
    out: Set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if p == "first":
            if n_paras > 0:
                out.add(0)
        elif p == "last":
            if n_paras > 0:
                out.add(n_paras - 1)
        else:
            try:
                one_based = int(p)
                if 1 <= one_based <= n_paras:
                    out.add(one_based - 1)
            except ValueError:
                continue
    return out


def apply_cn_boundaries(
    paragraphs: List[str],
    boundary_idx: Set[int],
    *,
    model,
    tokenizer,
    max_tokens: int,
    temperature: float,
) -> List[str]:
    result = []
    for pi, para in enumerate(paragraphs):
        if pi not in boundary_idx:
            result.append(para)
            continue
        sents = split_sentences(para)
        if not sents:
            result.append(para)
            continue

        def wrap(si: int) -> str:
            s = sents[si]
            if sentence_has_citation(s):
                return s
            zh = translate_en_to_cn_placeholder(
                model,
                tokenizer,
                s,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (
                "[中文边界 · 请人工回译为英文]\n"
                f"{zh}\n"
                "[英文原文 · 待替换]\n"
                f"{s}"
            )

        if len(sents) == 1:
            sents = [wrap(0)]
        else:
            sents = [wrap(0)] + sents[1:-1] + [wrap(len(sents) - 1)]
        result.append(" ".join(sents))
    return result


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "selections": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "selections" not in data or not isinstance(data["selections"], dict):
        data["selections"] = {}
    return data


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def interactive_pick(
    original: str,
    a: str,
    b: str,
    c: str,
    *,
    auto: Optional[int],
) -> Tuple[int, str]:
    if auto is not None:
        if auto == 0:
            return 0, original
        if auto == 1:
            return 1, a
        if auto == 2:
            return 2, b
        if auto == 3:
            return 3, c
    print("\n---")
    print("原句:", original)
    print("1:", a)
    print("2:", b)
    print("3:", c)
    print("0: 保留原句")
    while True:
        try:
            line = input("选择 0-3: ").strip()
        except EOFError:
            print("(EOF，默认保留原句)", file=sys.stderr)
            return 0, original
        if line not in ("0", "1", "2", "3"):
            print("请输入 0、1、2 或 3")
            continue
        choice = int(line)
        if choice == 0:
            return 0, original
        if choice == 1:
            return 1, a
        if choice == 2:
            return 2, b
        return 3, c


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Turnitout: interactive 3-way paraphrase per sentence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requires Apple Silicon Mac with >= 12 GB unified memory.
The base model (~5 GB) is downloaded automatically on first run.
""",
    )
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 English essay")
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-9b-it-4bit",
        help="MLX model ID or local path",
    )
    parser.add_argument("--adapter", default="adapter/", help="LoRA adapter directory")
    parser.add_argument("--state", type=Path, default=None, help="Checkpoint JSON for resume")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument(
        "--generation-mode",
        choices=["direct", "cn2en-bridge"],
        default="cn2en-bridge",
        help="Generation strategy: direct=EN paraphrase; cn2en-bridge=translate to CN then back to EN (recommended)",
    )
    parser.add_argument(
        "--auto",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Non-interactive: always pick 0=original 1/2/3=option",
    )
    parser.add_argument(
        "--cn-boundary-paras",
        default="",
        help="Comma-separated: first,last and/or 1-based para indices for CN boundary insertion",
    )
    parser.add_argument(
        "--boundary-only",
        action="store_true",
        help="Skip rewriting; only apply CN boundary blocks to existing text",
    )
    args = parser.parse_args()

    if mlx_load is None or mlx_generate is None:
        print("Error: mlx-lm not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    text = args.input.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(text)
    n_paras = len(paragraphs)

    print(f"Loading model {args.model!r} …", file=sys.stderr)
    if args.adapter:
        model, tokenizer = mlx_load(args.model, adapter_path=args.adapter)
    else:
        model, tokenizer = mlx_load(args.model)

    state: Dict[str, Any] = load_state(args.state) if args.state else {"version": 1, "selections": {}}
    selections: Dict[str, Any] = state.setdefault("selections", {})

    if args.boundary_only:
        bset = parse_boundary_paras(args.cn_boundary_paras, n_paras)
        if not bset:
            print("Error: --boundary-only requires --cn-boundary-paras", file=sys.stderr)
            return 1
        out_paras = apply_cn_boundaries(
            paragraphs,
            bset,
            model=model,
            tokenizer=tokenizer,
            max_tokens=min(args.max_tokens, 256),
            temperature=0.3,
        )
        args.output.write_text("\n\n".join(out_paras) + "\n", encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
        return 0

    built_paragraphs: List[List[str]] = []

    for pi, para in enumerate(paragraphs):
        sents = split_sentences(para)
        row: List[str] = []
        for si, sent in enumerate(sents):
            k = _key(pi, si)
            if k in selections:
                row.append(selections[k]["text"])
                continue

            if sentence_has_citation(sent):
                selections[k] = {"choice": -1, "text": sent, "skipped": "citation"}
                row.append(sent)
                if args.state:
                    save_state(args.state, state)
                continue

            a, b, c = generate_three_options(
                model,
                tokenizer,
                sent,
                max_tokens=args.max_tokens,
                base_temp=args.temperature,
                generation_mode=args.generation_mode,
            )
            choice, chosen = interactive_pick(sent, a, b, c, auto=args.auto)
            selections[k] = {"choice": choice, "text": chosen}
            row.append(chosen)
            if args.state:
                save_state(args.state, state)

        built_paragraphs.append(row)

    merged = [" ".join(sents) for sents in built_paragraphs]
    bset = parse_boundary_paras(args.cn_boundary_paras, len(merged))
    if bset:
        merged = apply_cn_boundaries(
            merged,
            bset,
            model=model,
            tokenizer=tokenizer,
            max_tokens=min(args.max_tokens, 256),
            temperature=0.3,
        )

    args.output.write_text("\n\n".join(merged) + "\n", encoding="utf-8")
    if args.state:
        state["output_paragraphs"] = merged
        save_state(args.state, state)
    print(f"Done. Written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
