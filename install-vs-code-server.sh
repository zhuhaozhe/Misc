cd ~/.vscode-server/bin
rm -rf ./${commit_id}/*
wget https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable
mv stable stable.tar.gz
tar -zxvf stable.tar.gz
mv vscode-server-linux-x64 ${commit_id}
