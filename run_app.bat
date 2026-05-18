@echo off
echo ============================================
echo  CAPTCHA Recognition App - Starting...
echo ============================================
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONIOENCODING=utf-8
streamlit run app.py --server.headless false
pause
