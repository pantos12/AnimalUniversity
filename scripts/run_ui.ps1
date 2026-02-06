param(
  [string]$Python = ".\\.venv\\Scripts\\python"
)

& $Python -m streamlit run apps/ui_streamlit/app.py
