import requests
import os
import getpass
import urllib
import json
from .utils import PROVISIONING_SCRIPT
from .exception import CashMoneyCarsException


def authenticate(token=None, debug=False):
    if not token:
        raise CashMoneyCarsException(("Provide token from "
                                      "https://dashboard.ngrok.com/auth"))

    result = os.popen(PROVISIONING_SCRIPT.format(token=token)).read()
    if debug:
        print(result)

    # Start daemon services. Need ipython so it can hang.
    get_ipython().system_raw('/usr/sbin/sshd -D &')
    get_ipython().system_raw(
        './ngrok start --config=/content/ngrok.yml --authtoken=$token --all &')
    get_ipython().system_raw('sleep 1')

    # Get ngrok url
    with urllib.request.urlopen(
            'http://localhost:4040/api/tunnels/first') as response:
        data = json.loads(response.read().decode())
        [host, port] = data['public_url'][6:].split(':')
        print(f'SSH command: ./sync.sh {port}')
