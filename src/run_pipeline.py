"""Driver: chạy lần lượt dvsktt_1_pretrain -> dvsktt_2_sft -> dvsktt_3_evaluate.

Nếu 1 notebook lỗi, in rõ: tên notebook, số cell (theo index và theo "In [N]"),
loại lỗi, và đoạn code của chính cell đó — không cần lục lại log dài của papermill.

Cách dùng trong 1 cell Colab (sau khi đã mount Drive + bật Notebook access cho
các secret KAGGLE_USERNAME / KAGGLE_KEY):

    !pip install -q papermill
    exec(open('/content/repo/src/run_pipeline.py').read())
"""

import logging
import os

import papermill as pm
from papermill.exceptions import PapermillExecutionError

# "unhandled iopub msg: colab_request" là log nhiễu vô hại của papermill khi
# gặp thông điệp nội bộ riêng của Colab — hạ mức log để không che mất lỗi thật.
logging.getLogger("papermill").setLevel(logging.CRITICAL)

NB_DIR = "/content/repo/notebooks"
NOTEBOOKS = [
    "dvsktt_1_pretrain.ipynb",
    "dvsktt_2_sft.ipynb",
    "dvsktt_3_evaluate.ipynb",
]


def run_pipeline(notebooks=NOTEBOOKS, nb_dir=NB_DIR):
    for nb_name in notebooks:
        path = os.path.join(nb_dir, nb_name)
        print(f"\n{'=' * 60}\n▶ Đang chạy: {nb_name}\n{'=' * 60}")
        try:
            pm.execute_notebook(path, path, kernel_name="python3", progress_bar=True)
        except PapermillExecutionError as e:
            print(f"\n{'!' * 60}")
            print(f"❌ LỖI tại notebook : {nb_name}")
            print(f"❌ Cell index       : {e.cell_index}  (In [{e.exec_count}])")
            print(f"❌ Loại lỗi         : {e.ename}: {e.evalue}")
            print(f"{'!' * 60}")
            print("\n--- Code của cell bị lỗi ---")
            print(e.source)
            print("\n--- Traceback đầy đủ ---")
            print("\n".join(e.traceback))
            raise
        else:
            print(f"✅ Hoàn thành: {nb_name}")

    print("\n🎉 Đã chạy xong cả 3 notebook thành công.")


if __name__ == "__main__":
    run_pipeline()
