#!/usr/bin/env python
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Author: liangxiaoping <shapeliang@gmail.com>
#
import client

def main():          
    func_name = "service_get_by_host_and_topic"
    kwargs = dict(host="node-54-148", topic="compute")
    body = {'args': [], 'kwargs': kwargs}

    try:
        result =  client.rpc_call(func_name, body=body)
        if result.get('host') == "node-54-148":
            print "==============OK=================="
        else:
            print "==============FAIL=================="
    except Exception:
        print "==============ERROR=================="


if __name__ == '__main__':
    main()