import socket
s = socket.socket()
s.settimeout(5)
r = s.connect_ex(("postgres", 5432))
print("OK" if r == 0 else "FAIL:" + str(r))
s.close()
