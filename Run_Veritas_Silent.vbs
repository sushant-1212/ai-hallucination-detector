Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Hp\OneDrive\Desktop\ai-hal"
WshShell.Run "python api.py", 0, False
