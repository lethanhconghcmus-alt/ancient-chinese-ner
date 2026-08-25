"""Tao ban NER SFT 4-loai (PER/LOC/DTM/TITLE) tu data 5-loai goc, gop ORG->LOC.

Muc dich: thu nghiem xem gop ORG vao LOC (giong quy uoc da dung o nhanh
TQ-merge cross-domain CRF, xem D:\\bio_source\\chisiec-ner\\scripts\\
merge_tq_ner_data.py) co giup Qwen SFT tren DVSKTT tot hon khong, va de
so sanh cong bang hon voi pretrain tren corpus TQ-merge (schema PER/LOC/OFI).

Dau ra: data/raw/ner_sft_4type/{train,dev,test}.jsonl — cung cau truc
instruction/input/output nhu data/raw/ner_sft, chi khac nhan ORG da doi
thanh LOC va instruction bo dong mo ta ORG.

Cach dung::

    python scripts/build_ner_sft_4type.py
"""
import json
import os
import re

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "raw", "ner_sft")
DST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "raw", "ner_sft_4type")

NEW_INSTRUCTION = (
    "你是一个专门处理越南汉文古籍的命名实体识别系统。请识别以下文本中的命名实体，"
    "并用{实体|类型}格式标注。\n实体类型：\n- PER: 人名 (人物姓名)\n"
    "- LOC: 地名 (地理位置，含机构/组织)\n- DTM: 时间 (日期时间)\n"
    "- TITLE: 官职 (官职称号)\n只输出标注后的文本，不要解释。"
)


def remap_output(output: str) -> str:
    # {entity|ORG} -> {entity|LOC}; giu nguyen cac loai khac.
    return re.sub(r"\{([^{}|]+)\|ORG\}", r"{\1|LOC}", output)


def convert(path_in: str, path_out: str) -> None:
    n_org = 0
    with open(path_in, encoding="utf-8") as f_in, open(path_out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            n_org += len(re.findall(r"\{[^{}|]+\|ORG\}", rec["output"]))
            rec["instruction"] = NEW_INSTRUCTION
            rec["output"] = remap_output(rec["output"])
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{os.path.basename(path_in)}: {n_org} ORG entities remapped -> LOC")


def main() -> None:
    os.makedirs(DST_DIR, exist_ok=True)
    for split in ("train", "dev", "test"):
        convert(os.path.join(SRC_DIR, f"{split}.jsonl"),
                os.path.join(DST_DIR, f"{split}.jsonl"))
    print(f"Done. Output: {DST_DIR}")


if __name__ == "__main__":
    main()
