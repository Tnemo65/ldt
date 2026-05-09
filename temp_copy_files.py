import io, tarfile, subprocess, sys
from pathlib import Path

PROJECT = Path("c:/proj/ldt")

for name, src, dst in [
    ('src', PROJECT / 'src', '/tmp/src'),
    ('models', PROJECT / 'models', '/tmp/models'),
]:
    files = [x for x in src.rglob('*') if x.is_file()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for item in files:
            arcname = str(item.relative_to(src))
            tar.add(item, arcname=arcname)
    buf.seek(0)
    p = subprocess.Popen(
        ['docker', 'exec', '-i', 'ldt-flink-jobmanager', 'tar', '-xf', '-', '-C', dst],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = p.communicate(input=buf.read())
    rc = p.returncode
    err_str = err.decode()[:200] if err else ''
    print(f'{name}: {len(files)} files, rc={rc}, err={err_str}')

print('Done copying files')
