"""Exercise the real HTML viewer in isolated system Chrome through CDP.

Optional developer validation uses the already installed aiohttp and Chrome;
neither is needed by the live server or offline viewer itself.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import time
import aiohttp


async def validate(url,output,profile,check_frame=None):
    port_file=profile/'DevToolsActivePort'
    for _ in range(100):
        if port_file.exists(): break
        await asyncio.sleep(.1)
    port=int(port_file.read_text().splitlines()[0])
    async with aiohttp.ClientSession(trust_env=False) as client:
        async with client.get(f'http://127.0.0.1:{port}/json/list') as response:
            page=next(x for x in await response.json() if x['type']=='page')
        async with client.ws_connect(page['webSocketDebuggerUrl']) as ws:
            sequence=0
            errors=[]
            network=[]
            async def cdp(method,params=None):
                nonlocal sequence
                sequence+=1
                await ws.send_json(dict(id=sequence,method=method,params=params or {}))
                while True:
                    try:
                        msg=await ws.receive_json(timeout=20)
                    except asyncio.TimeoutError:
                        print('CDP_TIMEOUT',method,json.dumps(network[-20:],ensure_ascii=False),flush=True)
                        raise
                    if msg.get('method')=='Runtime.exceptionThrown': errors.append(msg['params'])
                    if msg.get('method','').startswith('Network.'):
                        params=msg.get('params',{})
                        network.append(dict(method=msg['method'],url=params.get('request',params.get('response',{})).get('url'),
                                            error=params.get('errorText'),status=params.get('response',{}).get('status')))
                    if msg.get('id')==sequence:
                        if 'error' in msg: raise RuntimeError(msg['error'])
                        return msg.get('result',{})
            async def js(expression):
                result=await cdp('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
                if 'exceptionDetails' in result: raise RuntimeError(result['exceptionDetails'])
                return result['result'].get('value')
            await cdp('Page.enable');await cdp('Runtime.enable');await cdp('Network.enable')
            await cdp('Network.setCacheDisabled',dict(cacheDisabled=True))
            await cdp('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
            await cdp('Page.navigate',dict(url=url))
            for _ in range(100):
                result=await js('window.gsViewerTest ? window.gsViewerTest() : null')
                if result and result['loaded']>3 and result['lastDrawn'] is not None: break
                await asyncio.sleep(.1)
            assert result and result['loaded']>3,result
            count=result['loaded']
            await js("document.getElementById('follow').checked=false; document.getElementById('seek').value=3; document.getElementById('seek').dispatchEvent(new Event('input'))")
            await asyncio.sleep(.3)
            seek=await js('window.gsViewerTest()')
            assert seek['index']==3 and seek['lastDrawn']==3,seek
            await js("document.getElementById('play').click()")
            await asyncio.sleep(.8)
            playing=await js('window.gsViewerTest()')
            assert playing['index']>3,playing
            await js("document.getElementById('play').click();document.getElementById('seek').value=document.getElementById('seek').max;document.getElementById('seek').dispatchEvent(new Event('input'))")
            await asyncio.sleep(.4)
            last=await js('window.gsViewerTest()')
            assert last['index']==last['loaded']-1,last
            await js("document.getElementById('boxes').click()")
            await asyncio.sleep(.2)
            await js("document.getElementById('boxes').click()")
            if last.get('coverage'):
                await js("document.getElementById('seek').value=3;document.getElementById('seek').dispatchEvent(new Event('input'))")
                await asyncio.sleep(.3)
                rewind=await js('window.gsViewerTest()')
                assert rewind['coverage']['fraction']==seek['coverage']['fraction'],(rewind,seek)
                assert rewind['coverage']['fraction']<=last['coverage']['fraction']
                await js("document.getElementById('heatmap').click();document.getElementById('stations').click()")
                await asyncio.sleep(.1)
                await js("document.getElementById('heatmap').click();document.getElementById('stations').click();document.getElementById('seek').value=document.getElementById('seek').max;document.getElementById('seek').dispatchEvent(new Event('input'))")
                await asyncio.sleep(.2)
            checked=None
            if check_frame is not None:
                await js(f"document.getElementById('seek').value={check_frame};document.getElementById('seek').dispatchEvent(new Event('input'))")
                await asyncio.sleep(.4)
                checked=await js('window.gsViewerTest()')
                assert checked['index']==checked['lastDrawn']==check_frame,checked
            shot=await cdp('Page.captureScreenshot',dict(format='png',captureBeyondViewport=True))
            output.with_suffix('.png').write_bytes(base64.b64decode(shot['data']))
            assert not errors,errors
            report=dict(url=url,loaded_frames=count,seek=seek,playing=playing,last=last,javascript_errors=errors)
            if checked is not None: report['checked_frame']=checked
            if last.get('coverage'): report['coverage_rewind_no_future_evidence']=True
            output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('url');parser.add_argument('--output',required=True);parser.add_argument('--local-root');parser.add_argument('--check-frame',type=int);args=parser.parse_args()
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    server=None
    if args.local_root:
        from gs_search.visualization import LiveServer,SavedProvider
        server=LiveServer(args.local_root,SavedProvider(args.local_root),port=0)
        args.url=server.url
    with tempfile.TemporaryDirectory(prefix='gs-search-browser-',dir='/tmp') as directory:
        profile=Path(directory)
        with output.with_suffix('.chrome.log').open('w') as log:
            process=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-gpu',
                                      '--disable-dev-shm-usage','--disable-background-networking',
                                      '--no-proxy-server',
                                      '--password-store=basic','--use-mock-keychain',
                                      '--disable-features=HttpsUpgrades,HTTPSFirstModeV2',f'--log-net-log={profile/"netlog.json"}',
                                      f'--disk-cache-dir={profile/"http-cache"}',
                                      '--no-first-run','--disable-background-timer-throttling',
                                      '--remote-debugging-port=0',f'--user-data-dir={profile}','about:blank'],
                                     stdout=log,stderr=log)
            try: asyncio.run(validate(args.url,output,profile,args.check_frame))
            finally:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill();process.wait()
                if (profile/'netlog.json').exists(): shutil.copyfile(profile/'netlog.json',output.with_suffix('.netlog.json'))
                if server: server.close()
