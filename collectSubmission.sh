rm -f HW3.zip
zip -r HW3.zip . -x "*.git" "*cs231n/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" "*.pyc" "*cs231n/build/*" "*.zip"
