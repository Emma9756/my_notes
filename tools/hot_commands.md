
---
title: "hot_commands"
date: 2026-05-26
tags: 
  - tools
  - from_me
status: 更新中
---

## git
 ```bash
# set子仓
git submodule add git@github.com:Emma9756/work_notes.git work_notes
git submodule add git@github.com:Emma9756/secrets.git secrets

# get子仓
git submodule update --init --recursive

# 建议强制 .sh 使用 LF，否则 shell 脚本在 Git Bash / Linux 里可能出问题。
# 提交时把 CRLF 转成 LF，检出时不强制转成 CRLF
git config --global core.autocrlf input
 ```

## PowerShell
```bash
# 只对当前用户开启本地脚本权限
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\tools\sh_env.ps1

# 不用
  & "C:\Program Files\Git\bin\sh.exe" .\tools\commit_all.sh
# 使用
sh .\tools\commit_all.sh
```
 