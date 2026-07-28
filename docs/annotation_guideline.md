# Annotation Guideline — DVSKTT NER (v1.0)

Áp dụng cho việc duyệt gold test set (và mọi annotation về sau). Nguyên tắc
chung: **theo convention đa số đã có trong data v2** (để test không lệch quá xa
train silver), chỉ chuẩn hóa những chỗ mâu thuẫn. Metric là exact-match
`(surface, type)` nên boundary phải xác định được duy nhất từ guideline.

5 nhãn: `PER` 人名 · `LOC` 地名 · `ORG` 机构名 · `DTM` 时间 · `TITLE` 官职/tước vị.

---

## 1. PER — tên người

| Quy tắc | Ví dụ | Tag |
|---|---|---|
| Họ tên đầy đủ, tên riêng, tên húy | 鄭松, 吳士連, 祿續 | `{鄭松\|PER}` |
| Tên gọi tắt 1 ký tự khi rõ là người (sau họ đã nêu, hoặc vua gọi bề tôi) | 昌, 璉, 松 | `{昌\|PER}` |
| **Miếu hiệu / thụy hiệu chỉ đích danh một vua** | 太宗, 聖宗, 明宗, 太祖 | `{太宗\|PER}` |
| Chức + tên đi liền: **tách** chức và tên | 史臣吳士連 | `{史臣\|TITLE}{吳士連\|PER}` |
| KHÔNG tag: đại từ (朕, 帝 generic — xem TITLE), tên tự xưng chung (臣, 汝) | — | O |

## 2. TITLE — chức quan, tước vị, danh vị

| Quy tắc | Ví dụ | Tag |
|---|---|---|
| Chức quan, kể cả **cụm ghép bộ+chức lấy trọn** | 吏部尚書, 户部左侍郎, 監察御史 | `{吏部尚書\|TITLE}` |
| Tước phong đầy đủ (X王/公/侯/伯), **luôn TITLE kể cả khi dùng thay tên người** | 平安王, 郡公, 臨國公, 河汾伯 | `{平安王\|TITLE}` |
| Danh vị tôn xưng đứng riêng (theo convention đa số của data) | 帝, 上皇, 太后, 皇后, 太子, 皇帝 | `{帝\|TITLE}` |
| Danh vị khoa bảng | 進士, 同進士出身, 狀元 | `{進士\|TITLE}` |
| Chức thống lĩnh | 節制, 統領官, 都督 | `{節制\|TITLE}` |
| Tam công/tam thiếu | 太師, 太保, 太傅, 太尉, 少保 | `{太師\|TITLE}` |
| KHÔNG tag: danh từ chung không phải chức vị (功臣, 将臣, 士伍, 群臣) | — | O |

**Lưu ý boundary:** tước có tên đất phía trước lấy trọn (`平安王` chứ không tách
`平安`); chức ghép lấy trọn (`吏部尚書` chứ không `吏部`+`尚書`) — nhưng khi
`吏部` đứng MỘT MÌNH (cơ quan) thì là ORG.

## 3. ORG — cơ quan, tổ chức, triều đại/quốc hiệu

| Quy tắc | Ví dụ | Tag |
|---|---|---|
| Lục bộ, viện, giám, ty, vệ... đứng riêng | 吏部, 翰林院, 國子監, 錦衣衛 | `{吏部\|ORG}` |
| **Triều đại / quốc hiệu dùng như chính thể** (theo đa số data) | 宋, 明, 元, 漢, 莫, 越 | `{宋\|ORG}` |
| Quân đội của một phe | 官軍, 莫兵 | `{官軍\|ORG}` |
| 朝廷 (triều đình như một thể chế) | 朝廷 | `{朝廷\|ORG}` |
| KHÔNG tag: 天下, 國家 nghĩa chung | — | O |

## 4. LOC — địa danh

| Quy tắc | Ví dụ | Tag |
|---|---|---|
| Địa danh hành chính (xứ/phủ/huyện/châu/xã), lấy trọn cả hậu tố hành chính khi liền | 清華, 乂安, 鎮安府, 武陵社 | `{鎮安府\|LOC}` |
| Địa hình tự nhiên có tên riêng | 靈長海口, 西湖, 蘇瀝江 | `{西湖\|LOC}` |
| Kinh đô | 京師, 昇龍 | `{京師\|LOC}` |
| Nước ngoài với nghĩa lãnh thổ | 占城, 哀牢, 安南 | `{占城\|LOC}` |
| KHÔNG tag: 天下, phương hướng chung (西南, 内外) | — | O |

**Phân định ORG vs LOC cho tên nước/triều:** dùng như *chính thể/phe* (遣使如明,
宋兵) → ORG; dùng như *vùng đất* (入占城地) → LOC. Khi mơ hồ → ORG (theo đa số).

## 5. DTM — thời gian

| Quy tắc | Ví dụ | Tag |
|---|---|---|
| Cụm thời gian liên tục lấy TRỌN (mùa+tháng+ngày liền nhau) | 春正月, 冬十月十五日, 二十四日丑時 | `{冬十月十五日\|DTM}` |
| Can chi + số năm | 癸丑二年, 甲午 | `{癸丑二年\|DTM}` |
| Niên hiệu + năm (kể cả của TQ) — lấy trọn cụm | 洪德三年, 明成化十二年 | `{洪德三年\|DTM}` |
| Cụm can chi + niên hiệu ta + niên hiệu TQ đứng liền: tách theo từng đơn vị trọn nghĩa | 癸丑二年清康熙十二年 | `{癸丑二年\|DTM}{清康熙十二年\|DTM}` |
| KHÔNG tag: lượng thời gian (三年 = "ba năm" duration), 歲時, 日 chung | — | O |

**Lỗi hệ thống phải quét:** `春正` (49 lần, tag LOC) là `春正月` bị cắt — sửa
thành `{春正月|DTM}`.

## 6. Quy tắc chung

1. **Exact-match**: boundary phải trọn nghĩa — không tag nửa từ, không chứa
   dấu câu (`.`,`、`...).
2. **Không nest, không chồng lấn** — cụm dài hợp lệ thắng cụm con
   (`吏部尚書` TITLE, không tag thêm `吏部` bên trong).
3. **Mọi lần xuất hiện đều tag** (kể cả lặp trong cùng câu).
4. Chữ dị thể/khuyết (`󰱇`, `𭓇`...) nằm trong entity thì giữ nguyên trong surface.
5. Khi thực sự mơ hồ giữa 2 nhãn → ưu tiên theo bảng tần suất đa số của data
   (PER > LOC > ORG > TITLE > DTM không phải thứ tự ưu tiên — tra `docs/` note
   hoặc hỏi lại).

## Changelog

- v1.0 (2026-07-29): đúc từ pattern thực tế data v2. 4 quyết định chuẩn hóa
  đã được chốt (user duyệt 2026-07-29): miếu hiệu=PER · tước=TITLE
  (form-based, kể cả khi thay tên) · generic title (帝/上皇/太后...) giữ TITLE ·
  triều đại=ORG (LOC khi rõ nghĩa lãnh thổ). Kèm theo: 天下 bỏ,
  節制/進士=TITLE, 史臣吳士連 tách chức+tên, 春正→春正月 DTM.
