import os

def rename_files_remove_spaces():
    # 获取当前目录下的所有文件和文件夹（不递归）
    current_dir = os.getcwd()
    items = os.listdir(current_dir)

    # 遍历所有文件和文件夹
    for item in items:
        item_path = os.path.join(current_dir, item)

        # 仅处理文件，跳过文件夹
        if os.path.isfile(item_path):
            # 去掉文件名中的空格
            new_name = item.replace(" ", "")
            new_path = os.path.join(current_dir, new_name)

            # 重命名文件
            if new_name != item:  # 避免无意义重命名
                os.rename(item_path, new_path)
                print(f"Renamed: '{item}' -> '{new_name}'")
            else:
                print(f"No spaces in file name: '{item}'")

if __name__ == "__main__":
    rename_files_remove_spaces()
