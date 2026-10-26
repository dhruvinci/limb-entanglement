"""Loopback-only review server with HTTP byte ranges for seeking the cached master."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def byte_range(value,size):
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',value.strip())
    if not match or size<=0:
        raise ValueError('Invalid range')
    start,end=match.groups()
    if not start:
        if not end or int(end)<=0: raise ValueError('Invalid suffix')
        return max(0,size-int(end)),size-1
    start=int(start);end=min(size-1,int(end)) if end else size-1
    if not 0<=start<=end<size: raise ValueError('Unsatisfiable range')
    return start,end


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.active_range=None
        path=Path(self.translate_path(self.path))
        requested=self.headers.get('Range')
        if not requested or not path.is_file():
            return super().send_head()
        try:
            stream=path.open('rb')
        except OSError:
            self.send_error(404);return None
        size=path.stat().st_size
        try:
            start,end=byte_range(requested,size)
        except ValueError:
            stream.close();self.send_response(416)
            self.send_header('Content-Range',f'bytes */{size}')
            self.send_header('Content-Length','0');self.end_headers();return None
        self.active_range=(start,end)
        stream.seek(start)
        self.send_response(206)
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(end-start+1))
        self.end_headers()
        return stream

    def end_headers(self):
        self.send_header('Accept-Ranges','bytes')
        if self.path.endswith(('.html','/')):
            self.send_header('Cache-Control','no-cache')
        super().end_headers()

    def copyfile(self,source,outputfile):
        try:
            if self.active_range is None:
                return super().copyfile(source,outputfile)
            remaining=self.active_range[1]-self.active_range[0]+1
            while remaining:
                block=source.read(min(256*1024,remaining))
                if not block: break
                outputfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):
            pass  # Browser legitimately cancels the old request when seeking.


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(RangeHandler,directory=str(ROOT)))
    print(f'Pilot review: http://127.0.0.1:{args.port}/review/',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=='__main__':
    main()
