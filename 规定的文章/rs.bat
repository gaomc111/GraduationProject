@echo off
setlocal enabledelayedexpansion

:: 遍历当前目录下的所有文件
for %%f in (*) do (
    :: 原始文件名
    set "filename=%%f"
    
    :: 替换文件名中的空格为无空格形式
    set "newname=!filename: =!"
    
    :: 如果新旧文件名不同，则重命名
    if not "!filename!"=="!newname!" (
        ren "%%f" "!newname!"
        echo Renamed: "%%f" -> "!newname!"
    ) else (
        echo No spaces in file name: "%%f"
    )
)

pause
