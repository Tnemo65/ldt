import os, signal, psutil
killed = 0
for p in psutil.process_iter(['pid', 'name']):
    try:
        if p.info['name'] == 'python' and p.info['pid'] != os.getpid():
            os.kill(p.info['pid'], signal.SIGTERM)
            killed += 1
    except:
        pass
print(f'Killed {killed} stale processes')
