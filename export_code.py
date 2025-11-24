import os

# 设置要忽略的文件夹（比如虚拟环境、git目录、结果文件夹）
IGNORE_DIRS = {'.git', '__pycache__', 'results', 'venv', '.idea', '.vscode'}
# 设置输出文件名
OUTPUT_FILE = 'project_code_context.txt'


def merge_files():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # 遍历当前目录
        for root, dirs, files in os.walk("."):
            # 过滤掉不需要的目录
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    # 写入分隔符和文件名，方便AI识别
                    outfile.write(f"\n{'=' * 20}\nFILE: {file_path}\n{'=' * 20}\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"# Error reading file: {e}")
                    outfile.write("\n")

    print(f"完成！所有代码已合并到 {OUTPUT_FILE}，请将该文件发送给AI。")


if __name__ == "__main__":
    merge_files()