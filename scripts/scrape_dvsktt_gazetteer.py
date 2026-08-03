"""Scrape toàn văn Đại Việt Sử Ký Toàn Thư (bản khắc Chính Hoà 1697) từ
nomfoundation.org để xây gazetteer tên riêng (người / đất / sách / quan chức /
niên hiệu) phục vụ NER trên Hán văn.

nomfoundation.org không có XML/API export — dữ liệu render server-side
(PHP + DB) thành HTML thuần, phân trang qua POST field ``curPg``. Script này
scrape trực tiếp HTML đó.

Mỗi trang gồm 2 phần:
  - "Split sentence and Phonetics": từng câu Hán văn gốc (mỗi chữ 1 thẻ
    <font>, có toạ độ [trang*cột*chữ]) kèm phiên âm Hán-Việt.
  - "Vietnamese Translation": bản dịch tiếng Việt, một số cụm từ có chú
    thích dạng "Tên: giải nghĩa" qua tooltip (ONMOUSEOVER).

Vì phiên âm Hán-Việt về cơ bản là 1 âm tiết = 1 chữ Hán, script align
ngược tên riêng (dạng Hán-Việt trong chú thích) về đúng chuỗi chữ Hán bằng
cách so khớp âm tiết với câu phiên âm cùng mục — best-effort, không phải
lúc nào cũng khớp được (``term_han`` sẽ là null nếu không tìm ra).

Cách dùng::

    py scripts/scrape_dvsktt_gazetteer.py --out data/raw/gazetteer
    py scripts/scrape_dvsktt_gazetteer.py --sections 102-Ngoai-ky-toan-thu --out /tmp/test
"""

import argparse
import json
import os
import re
import time
from html import unescape
from typing import Dict, Iterator, List, Optional

import requests

BASE_URL = 'https://nomfoundation.org/nom-project/history-of-greater-vietnam/Fulltext'
HEADERS = {
    'User-Agent': 'DVSKTT-NER-research-scraper/1.0 '
                  '(academic NER gazetteer building; contact lethanhcong.hcmus@gmail.com)'
}

# Thứ tự các mục (quyển) trong Fulltext, lấy từ menu điều hướng của trang
# History-of-Greater-Vietnam/Fulltext (menu PHP Layers Menu, các layer L1..L81).
SECTION_SLUGS: List[str] = [
    '1-Ky-Hong-Bang-thi', '2-Ky-nha-Thuc', '3-Ky-nha-Trieu', '4-Ky-thuoc-Tay-Han',
    '5-Ky-Trung-Nu-Vuong', '6-Ky-thuoc-Dong-Han', '7-Ky-Si-Vuong',
    '8-Ky-thuoc-Ngo-Tan-Tong-Te', '9-Ky-tien-Ly', '10-Ky-Trieu-Viet-Vuong',
    '11-Ky-hau-Ly', '12-Ky-thuoc-Tuy-Duong', '13-Ky-Nam-Bac-phan-tranh',
    '14-Ky-nha-Ngo', '24-Ky-nha-Dinh', '26-Ky-nha-Le', '29-Thai-To-Hoang-De',
    '30-Thai-Tong-Hoang-De', '31-Thanh-Tong-Hoang-De', '32-Nhan-Tong-Hoang-De',
    '33-Than-Tong-Hoang-De', '34-Anh-Tong-Hoang-De', '35-Cao-Tong-Hoang-De',
    '36-Hue-Tong-Hoang-De', '37-Chieu-Hoang', '38-Thai-Tong-Hoang-De',
    '39-Thanh-Tong-Hoang-De', '40-Nhan-Tong-Hoang-De', '41-Anh-Tong-Hoang-De',
    '42-Minh-Tong-Hoang-De', '43-Hien-Tong-Hoang-De', '44-Du-Tong-Hoang-De',
    '46-Nghe-Tong-Hoang-De', '48-Due-Tong-Hoang-De', '49-Phe-De',
    '50-Thuan-Tong-Hoang-De', '51-Thieu-De', '52-Phu-Ho-Quy-Ly-Ho-Han-Thuong',
    '54-Ky-hau-Tran', '56-Ky-thuoc-Minh', '57-Thai-Tong-Cao-Hoang-De',
    '58-Thanh-Tong-Van-Hoang-De', '59-Nhan-Tong-Tuyen-Hoang-De',
    '60-Thanh-Tong-Thuan-Hoang-De-thuong', '61-Thanh-Tong-Thuan-Hoang-De-ha',
    '62-Hien-Tong-Due-Hoang-De', '63-Tuc-Tong-Kham-Hoang-De', '64-Uy-Muc-De',
    '65-Tuong-Duc-De', '66-Chieu-Tong-Than-Hoang-De', '67-Cung-Hoang-De',
    '68-Phu-Mac-Dang-Dung-Mac-Dang-Doanh', '70-Trang-Tong-Du-Hoang-De',
    '71-Phu-Mac-Dang-Doanh-Mac-Phuc-Nguyen', '74-Trung-Tong-Vu-Hoang-De',
    '75-Phu-Mac-Phuc-Nguyen', '76-Anh-Tong-Tuan-Hoang-De', '77-Phu-Mac-Phuc-Nguyen',
    '79-The-Tong-Nghi-Hoang-De', '80-Phu-Mac-Hau-Hop', '81-Kinh-Tong-Hue-Hoang-De',
    '82-Than-Tong-Uyen-Hoang-De-thuong', '83-Chan-Tong-Thuan-Hoang-De',
    '84-Than-Tong-Uyen-Hoang-De-ha', '85-Huyen-Tong-Muc-Hoang-De',
    '86-Gia-Tong-My-Hoang-De', '100-Tuc-bien-tu', '101-Tuc-bien-thu',
    '102-Ngoai-ky-toan-thu', '103-Toan-thu-bieu', '104-Toan-thu-pham-le',
    '105-Ky-nien-muc-luc', '106-Khao-tong-luan',
]

PAGE_COUNT_RE = re.compile(r'\[\s*(\d+)\s*pages?\s*\]')

SENTENCE_RE = re.compile(
    r'<span class=hnText>(?P<han>.*?)</span>'
    r"<a[^>]*ONMOUSEOVER=\"return escape\('\[Page \* column \* character\]'\);\"[^>]*>"
    r'<font[^>]*>\[(?P<pos>[^\]]+)\]</font></a><br>'
    r'(?P<phienam>[^<]*)<br>',
    re.DOTALL,
)
CHAR_RE = re.compile(r'<font size = 4>\s*(\S)\s*</font>')
GLOSS_RE = re.compile(r"ONMOUSEOVER=\"return escape\('(.*?)'\);\"", re.DOTALL)
TOKEN_RE = re.compile(r"[^\s,\.;:\-–—]+")

# Heuristic gán nhãn loại thực thể theo từ khoá trong phần giải nghĩa.
# Không chính xác tuyệt đối — chỉ để phân loại sơ bộ, cần người review lại.
TYPE_RULES = [
    (('sách', 'sử ', 'kinh ', 'truyện', 'thi tập', 'bộ sử'), 'ORG'),
    (('núi', 'sông', 'huyện', 'châu ', 'phủ ', 'trấn ', 'xã ', 'làng ',
      'thành ', 'cửa ', 'đất ', 'nước ', 'kinh đô'), 'LOC'),
    (('niên hiệu', 'triều đại', 'đời vua'), 'DTM'),
    (('chức quan', 'quan chức', 'tước ', 'phẩm hàm'), 'TITLE'),
]


def guess_type(definition: str) -> str:
    low = definition.lower()
    for keywords, label in TYPE_RULES:
        if any(k in low for k in keywords):
            return label
    return 'PER'  # mặc định: phần lớn chú thích còn lại là tên người


def fetch_page(session: requests.Session, slug: str, page_idx: int,
               retries: int = 2) -> str:
    url = f'{BASE_URL}/{slug}?uiLang=en'
    # (connect, read) riêng biệt — tránh treo vô hạn nếu server phản hồi
    # nhỏ giọt (chunked) mà không bao giờ chạm read timeout tổng.
    timeout = (10, 20)
    last_err = None
    for attempt in range(retries + 1):
        try:
            if page_idx == 0:
                resp = session.get(url, headers=HEADERS, timeout=timeout)
            else:
                resp = session.post(url, data={'curPg': str(page_idx)},
                                     headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last_err


def get_total_pages(html: str) -> int:
    m = PAGE_COUNT_RE.search(html)
    return int(m.group(1)) if m else 1


def parse_sentences(html: str, slug: str, page_idx: int) -> Iterator[Dict]:
    for m in SENTENCE_RE.finditer(html):
        han_chars = CHAR_RE.findall(m.group('han'))
        if not han_chars:
            continue
        yield {
            'section': slug,
            'page_idx': page_idx,
            'position': m.group('pos'),
            'han': ''.join(han_chars),
            'han_chars': han_chars,
            'phienam': unescape(m.group('phienam')).strip(),
        }


def parse_terms(html: str, slug: str, page_idx: int) -> Iterator[Dict]:
    for m in GLOSS_RE.finditer(html):
        raw = unescape(m.group(1)).strip()
        if ':' not in raw:
            continue  # loại các tooltip không phải chú thích tên riêng
        term, _, definition = raw.partition(':')
        term, definition = term.strip(), definition.strip()
        if not term or not definition or len(term) > 40:
            continue
        yield {
            'term_hanviet': term,
            'definition': definition,
            'type_guess': guess_type(definition),
            'section': slug,
            'page_idx': page_idx,
        }


def align_term_to_han(term: str, sentences: List[Dict]) -> Optional[str]:
    """Best-effort: khớp âm tiết Hán-Việt của `term` vào câu phiên âm cùng
    mục, rồi lấy đúng chuỗi chữ Hán ở cùng vị trí (1 âm tiết ~ 1 chữ Hán).
    Trả None nếu không tìm được (số âm tiết lệch số chữ, hoặc không khớp).
    """
    term_syll = [t.lower() for t in TOKEN_RE.findall(term)]
    n = len(term_syll)
    if n == 0:
        return None
    for sent in sentences:
        syll = [t.lower() for t in TOKEN_RE.findall(sent['phienam'])]
        chars = sent['han_chars']
        if len(syll) != len(chars):
            continue  # dấu câu lệch giữa phiên âm và số chữ Hán -> bỏ qua
        for i in range(len(syll) - n + 1):
            if syll[i:i + n] == term_syll:
                return ''.join(chars[i:i + n])
    return None


def scrape_section(session: requests.Session, slug: str, delay: float):
    html = fetch_page(session, slug, 0)
    total_pages = get_total_pages(html)
    print(f'  {total_pages} trang')

    sentences: List[Dict] = []
    terms: List[Dict] = []
    for page_idx in range(total_pages):
        if page_idx > 0:
            time.sleep(delay)
            html = fetch_page(session, slug, page_idx)
        sentences.extend(parse_sentences(html, slug, page_idx))
        terms.extend(parse_terms(html, slug, page_idx))

    for t in terms:
        t['term_han'] = align_term_to_han(t['term_hanviet'], sentences)

    return sentences, terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/raw/gazetteer',
                        help='Thư mục output (mặc định: data/raw/gazetteer)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Giây nghỉ giữa các request (mặc định: 1.0)')
    parser.add_argument('--sections', nargs='*', default=None,
                        help='Chỉ scrape 1 số mục (debug); mặc định: toàn bộ 73 mục')
    parser.add_argument('--append', action='store_true',
                        help='Nối vào file output có sẵn thay vì ghi đè (dùng để resume)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    slugs = args.sections or SECTION_SLUGS

    sent_path = os.path.join(args.out, 'dvsktt_sentences.jsonl')
    term_path = os.path.join(args.out, 'dvsktt_terms.jsonl')

    session = requests.Session()
    n_sent = n_term = n_matched = 0
    mode = 'a' if args.append else 'w'

    with open(sent_path, mode, encoding='utf-8') as fs, \
         open(term_path, mode, encoding='utf-8') as ft:
        for i, slug in enumerate(slugs, 1):
            print(f'[{i}/{len(slugs)}] {slug}', flush=True)
            try:
                sentences, terms = scrape_section(session, slug, args.delay)
            except requests.RequestException as e:
                print(f'  LOI {slug}: {e}', flush=True)
                continue

            for s in sentences:
                fs.write(json.dumps(s, ensure_ascii=False) + '\n')
            for t in terms:
                ft.write(json.dumps(t, ensure_ascii=False) + '\n')
                n_matched += t['term_han'] is not None
            fs.flush()
            ft.flush()

            n_sent += len(sentences)
            n_term += len(terms)
            print(f'  +{len(sentences)} cau, +{len(terms)} term', flush=True)
            time.sleep(args.delay)

    print(f'\nXong. {n_sent} cau moi -> {sent_path}')
    print(f'      {n_term} term moi ({n_matched} khop duoc chu Han) -> {term_path}')


if __name__ == '__main__':
    main()
