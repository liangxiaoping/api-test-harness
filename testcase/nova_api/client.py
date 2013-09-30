# Copyright 2012 OpenStack LLC.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import sys
import copy
import httplib
import posixpath
import socket
import StringIO
import urlparse
import functools
import inspect
import itertools
import xmlrpclib
import datetime
import json
import logging

from oslo.config import cfg
from simpleauth.signature import AccessToken

DEFAULT_LOG_FORMAT = '%(asctime)s.%(msecs)03d %(levelname)s %(thread)d %(module)s %(funcName)s %(message)s'
#logging.basicConfig(format=DEFAULT_LOG_FORMAT, level=logging.DEBUG, filename='harness.log')
logging.basicConfig(format=DEFAULT_LOG_FORMAT, level=logging.DEBUG, stream=sys.stdout)

LOG = logging.getLogger(__name__)


USER_AGENT = 'harness'
CHUNKSIZE = 1024 * 64  # 64kB

brood_opts = [
    cfg.StrOpt('test_endpoint',
               default= '10.23.54.151:19595' ,
               help='host ip on where the test endpoint python module is running'),
    cfg.IntOpt('http_retry_times',
               default=3,
               help="http request retry times"),
    cfg.BoolOpt('log_http',
               default=False,
               help="Log http request or response"),
]

CONF = cfg.CONF
CONF.register_opts(brood_opts)

DATETIME_KEYS = ('created_at', 'deleted_at', 'updated_at', 'launched_at',
                 'scheduled_at', 'terminated_at', 'attach_time', 'expire',
                 'start_period', 'last_refreshed','instance_updated',
                 'instance_created',
)


DEFAULT_CONTEXT = {'project_name': None, 'event_type': None,
                   'timestamp': None, 'auth_token': None,
                   'remote_address': None, 'quota_class': None,
                   'is_admin': True, 'service_catalog': None,
                   'read_deleted': u'no', 'user_id': None,
                   'roles': [],'request_id': None,
                   'instance_lock_checked': False, 'project_id': None,
                   'user_name': None}

def rpc_call(func_name, context=DEFAULT_CONTEXT, body=None):
    return do_request(func_name, context, body)

def convert_datetimes(values, *datetime_keys):
    for key in values:
        if key in datetime_keys and isinstance(values[key], basestring):
            values[key] = parse_strtime(values[key])
    return values

def convert_results(values, to_datetime=True, level=0, max_depth=3):
    if level > max_depth:
        return '?'

    try:
        recursive = functools.partial(convert_results,
                                      to_datetime=to_datetime,
                                      level=level,
                                      max_depth=max_depth)

        if isinstance(values, (list, tuple)):
            return [recursive(v) for v in values]
        elif isinstance(values, dict):
            values = convert_datetimes(values, *DATETIME_KEYS)
            return AttrDict((k, recursive(v)) for k, v in values.iteritems())
        elif hasattr(values, 'iteritems'):
            return recursive(dict(values.iteritems()), level=level + 1)
        elif hasattr(values, '__iter__'):
            return recursive(list(values))
        else :
            return values

    except TypeError:
        return values

class AttrDict(dict):
    """
    http://stackoverflow.com/questions/4984647/\
    accessing-dict-keys-like-an-attribute-in-python
    http://bugs.python.org/issue1469629
    """
    def __init__(self, source):
        super(AttrDict, self).__init__(source)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

def do_request(func_name, context, body=None, method='POST'):
    headers = {}

    #context_dict = {"nova_context" : context.to_dict()}
    context_dict = {"nova_context" : context}
    if body:
        body = dict(context_dict, **body)
    else:
        body = context_dict

    kwargs = {}
    kwargs['headers'] = headers
    kwargs['body'] = body

    url = "v1/db/nova/%s" % func_name
    client = HTTPClient(CONF.test_endpoint)
    for i in xrange(CONF.http_retry_times):
        try:
            resp, resp_body = client.json_request(method, url, **kwargs)
            if 'result' in resp_body:
                return convert_results(resp_body.get('result', None))
            else:
                LOG.debug("Exception[%s] raised." % resp_body.get('exception'))
                raise
        except Exception as e :
            LOG.error("http request %s failed %d times: %s" %
                      (url, i, unicode(e)))

class HTTPClient(object):

    def __init__(self, endpoint, **kwargs):
        #endpoint must start with 'http', or urlparse will not work rightly
        if not endpoint.startswith('http'):
            endpoint = 'http://' + endpoint
        self.endpoint = endpoint
        endpoint_parts = self.parse_endpoint(self.endpoint)
        self.endpoint_scheme = endpoint_parts.scheme
        self.endpoint_hostname = endpoint_parts.hostname
        self.endpoint_port = endpoint_parts.port
        self.endpoint_path = endpoint_parts.path

        self.connection_class = self.get_connection_class(self.endpoint_scheme)
        self.connection_kwargs = self.get_connection_kwargs(
            self.endpoint_scheme, **kwargs)


    @staticmethod
    def parse_endpoint(endpoint):
        return urlparse.urlparse(endpoint)

    @staticmethod
    def get_connection_class(scheme):
        if scheme == 'https':
            return None
        else:
            return httplib.HTTPConnection

    @staticmethod
    def get_connection_kwargs(scheme, **kwargs):
        _kwargs = {'timeout': float(kwargs.get('timeout', 120))}

        if scheme == 'https':
            _kwargs['cacert'] = kwargs.get('cacert', None)
            _kwargs['cert_file'] = kwargs.get('cert_file', None)
            _kwargs['key_file'] = kwargs.get('key_file', None)
            _kwargs['insecure'] = kwargs.get('insecure', False)
            _kwargs['ssl_compression'] = kwargs.get('ssl_compression', True)

        return _kwargs

    def get_connection(self):
        _class = self.connection_class
        if not _class:
           raise
        try:
            return _class(self.endpoint_hostname, self.endpoint_port,
                          **self.connection_kwargs)
        except httplib.InvalidURL:
            raise

    def log_curl_request(self, method, url, kwargs):
        if not CONF.log_http:
            return

        curl = ['curl -i -X %s' % method]

        for (key, value) in kwargs['headers'].items():
            header = '-H \'%s: %s\'' % (key, value)
            curl.append(header)

        conn_params_fmt = [
            ('key_file', '--key %s'),
            ('cert_file', '--cert %s'),
            ('cacert', '--cacert %s'),
        ]
        for (key, fmt) in conn_params_fmt:
            value = self.connection_kwargs.get(key)
            if value:
                curl.append(fmt % value)

        if self.connection_kwargs.get('insecure'):
            curl.append('-k')

        if 'body' in kwargs:
            curl.append('-d \'%s\'' % kwargs['body'])

        curl.append('%s/%s' % (self.endpoint, url))
        LOG.debug(ensure_str(' '.join(curl)))

    @staticmethod
    def log_http_response(resp, body=None):
        if not CONF.log_http:
            return

        status = (resp.version / 10.0, resp.status, resp.reason)
        dump = ['\nHTTP/%.1f %s %s' % status]
        dump.extend(['%s: %s' % (k, v) for k, v in resp.getheaders()])
        dump.append('')
        if body:
            dump.extend([body, ''])
        LOG.debug(ensure_str('\n'.join(dump)))

    @staticmethod
    def encode_headers(headers):
        """
        Encodes headers.

        Note: This should be used right before
        sending anything out.

        :param headers: Headers to encode
        :returns: Dictionary with encoded headers'
                  names and values
        """
        to_str = ensure_str
        return dict([(to_str(h), to_str(v)) for h, v in headers.iteritems()])

    def _http_request(self, url, method, **kwargs):
        """ Send an http request with the specified characteristics.

        Wrapper around httplib.HTTP(S)Connection.request to handle tasks such
        as setting headers and error handling.
        """
        # Copy the kwargs so we can reuse the original in case of redirects
        kwargs['headers'] = copy.deepcopy(kwargs.get('headers', {}))
        kwargs['headers'].setdefault('User-Agent', USER_AGENT)

        self.log_curl_request(method, url, kwargs)
        conn = self.get_connection()
        # Note(flaper87): Before letting headers / url fly,
        # they should be encoded otherwise httplib will
        # complain. If we decide to rely on python-request
        # this wont be necessary anymore.
        kwargs['headers'] = self.encode_headers(kwargs['headers'])

        try:
            conn_url = posixpath.normpath('%s/%s' % (self.endpoint_path, url))
            # Note(flaper87): Ditto, headers / url
            # encoding to make httplib happy.
            conn_url = ensure_str(conn_url)
            if kwargs['headers'].get('Transfer-Encoding') == 'chunked':
                conn.putrequest(method, conn_url)
                for header, value in kwargs['headers'].items():
                    conn.putheader(header, value)
                conn.endheaders()
                chunk = kwargs['body'].read(CHUNKSIZE)
                # Chunk it, baby...
                while chunk:
                    conn.send('%x\r\n%s\r\n' % (len(chunk), chunk))
                    chunk = kwargs['body'].read(CHUNKSIZE)
                conn.send('0\r\n\r\n')
            else:
                conn.request(method, conn_url, **kwargs)
            resp = conn.getresponse()
        except socket.gaierror as e:
            message = "Error finding address for %s: %s" % (
                self.endpoint_hostname, e)
            raise
        except (socket.error, socket.timeout) as e:
            endpoint = self.endpoint
            message = "Error communicating with %(endpoint)s %(e)s" % locals()
            raise

        body_iter = ResponseBodyIterator(resp)
        # LOG.debug("content-type:%s" % resp.getheader('content-type'))

        # Read body into string if it isn't obviously image data

        if 'application/octet-stream' not in resp.getheader('content-type', None):
            body_str = ''.join([chunk for chunk in body_iter])
            self.log_http_response(resp, body_str)
            body_iter = StringIO.StringIO(body_str)
        else:
            self.log_http_response(resp)

        LOG.info("%s returned with HTTP %d" % (url, resp.status))

        if 400 <= resp.status < 600:
            LOG.error("Request returned failure status.")
            raise
        elif resp.status in (301, 302, 305):
            # Redirected. Reissue the request to the new location.
            return self._http_request(resp['location'], method, **kwargs)
        elif resp.status == 300:
            raise

        return resp, body_iter

    def json_request(self, method, url, **kwargs):
        kwargs.setdefault('headers', {})
        kwargs['headers'].setdefault('Content-Type', 'application/json')
        kwargs['headers'].setdefault('Accept', 'application/json')

        if 'body' in kwargs:
            kwargs['body'] = json.dumps(to_primitive(kwargs['body'],
                                                     convert_instances=True))

        resp, body_iter = self._http_request(url, method, **kwargs)

        if 'application/json' in resp.getheader('content-type', None):
            body = ''.join([chunk for chunk in body_iter])
            try:
                body = json.loads(body)
            except ValueError:
                LOG.error('Could not decode response body as JSON')
        else:
            body = ''.join([chunk for chunk in body_iter])
        return  resp, body

    def raw_request(self, method, url, **kwargs):
        kwargs.setdefault('headers', {})
        kwargs['headers'].setdefault('Content-Type',
                                     'application/octet-stream')
        kwargs['headers'].setdefault('Accept', 'application/octet-stream')

        if 'body' in kwargs:
            if (hasattr(kwargs['body'], 'read')
                    and method.lower() in ('post', 'put')):
                # We use 'Transfer-Encoding: chunked' because
                # body size may not always be known in advance.
                kwargs['headers']['Transfer-Encoding'] = 'chunked'
        return self._http_request(url, method, **kwargs)

class ResponseBodyIterator(object):
    """A class that acts as an iterator over an HTTP response."""

    def __init__(self, resp):
        self.resp = resp

    def __iter__(self):
        while True:
            yield self.next()

    def next(self):
        chunk = self.resp.read(CHUNKSIZE)
        if chunk:
            return chunk
        else:
            raise StopIteration()

def ensure_unicode(text, incoming=None, errors='strict'):
    """
    Decodes incoming objects using `incoming` if they're
    not already unicode.

    :param incoming: Text's current encoding
    :param errors: Errors handling policy.
    :returns: text or a unicode `incoming` encoded
                representation of it.
    """
    if isinstance(text, unicode):
        return text

    if not incoming:
        incoming = sys.stdin.encoding or \
            sys.getdefaultencoding()

    # Calling `str` in case text is a non str
    # object.
    text = str(text)
    try:
        return text.decode(incoming, errors)
    except UnicodeDecodeError:
        # Note(flaper87) If we get here, it means that
        # sys.stdin.encoding / sys.getdefaultencoding
        # didn't return a suitable encoding to decode
        # text. This happens mostly when global LANG
        # var is not set correctly and there's no
        # default encoding. In this case, most likely
        # python will use ASCII or ANSI encoders as
        # default encodings but they won't be capable
        # of decoding non-ASCII characters.
        #
        # Also, UTF-8 is being used since it's an ASCII
        # extension.
        return text.decode('utf-8', errors)


def ensure_str(text, incoming=None,
               encoding='utf-8', errors='strict'):
    """
    Encodes incoming objects using `encoding`. If
    incoming is not specified, text is expected to
    be encoded with current python's default encoding.
    (`sys.getdefaultencoding`)

    :param incoming: Text's current encoding
    :param encoding: Expected encoding for text (Default UTF-8)
    :param errors: Errors handling policy.
    :returns: text or a bytestring `encoding` encoded
                representation of it.
    """

    if not incoming:
        incoming = sys.stdin.encoding or \
            sys.getdefaultencoding()

    if not isinstance(text, basestring):
        # try to convert `text` to string
        # This allows this method for receiving
        # objs that can be converted to string
        text = str(text)

    if isinstance(text, unicode):
        return text.encode(encoding, errors)
    elif text and encoding != incoming:
        # Decode text before encoding it with `encoding`
        text = ensure_unicode(text, incoming, errors)
        return text.encode(encoding, errors)

    return text


# ISO 8601 extended time format with microseconds
_ISO8601_TIME_FORMAT_SUBSECOND = '%Y-%m-%dT%H:%M:%S.%f'
_ISO8601_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
PERFECT_TIME_FORMAT = _ISO8601_TIME_FORMAT_SUBSECOND

def strtime(at=None, fmt=PERFECT_TIME_FORMAT):
    """Returns formatted utcnow."""
    if not at:
        at = datetime.datetime.utcnow()
    return at.strftime(fmt)

def parse_strtime(timestr, fmt=PERFECT_TIME_FORMAT):
    """Turn a formatted time back into a datetime."""
    return datetime.datetime.strptime(timestr, fmt)


def to_primitive(value, convert_instances=False, level=0):
    """Convert a complex object into primitives.

    Handy for JSON serialization. We can optionally handle instances,
    but since this is a recursive function, we could have cyclical
    data structures.

    To handle cyclical data structures we could track the actual objects
    visited in a set, but not all objects are hashable. Instead we just
    track the depth of the object inspections and don't go too deep.

    Therefore, convert_instances=True is lossy ... be aware.

    """
    nasty = [inspect.ismodule, inspect.isclass, inspect.ismethod,
             inspect.isfunction, inspect.isgeneratorfunction,
             inspect.isgenerator, inspect.istraceback, inspect.isframe,
             inspect.iscode, inspect.isbuiltin, inspect.isroutine,
             inspect.isabstract]
    for test in nasty:
        if test(value):
            return unicode(value)

    # value of itertools.count doesn't get caught by inspects
    # above and results in infinite loop when list(value) is called.
    if type(value) == itertools.count:
        return unicode(value)

    if getattr(value, '__module__', None) == 'mox':
        return 'mock'

    if level > 4:
        return '?'

    # The try block may not be necessary after the class check above,
    # but just in case ...
    try:
        # It's not clear why xmlrpclib created their own DateTime type, but
        # for our purposes, make it a datetime type which is explicitly
        # handled
        if isinstance(value, xmlrpclib.DateTime):
            value = datetime.datetime(*tuple(value.timetuple())[:6])

        if isinstance(value, (list, tuple)):
            o = []
            for v in value:
                o.append(to_primitive(v, convert_instances=convert_instances,
                                      level=level))
            return o
        elif isinstance(value, dict):
            o = {}
            for k, v in value.iteritems():
                o[k] = to_primitive(v, convert_instances=convert_instances,
                                    level=level)
            return o
        elif isinstance(value, datetime.datetime):
            return strtime(value)
        elif hasattr(value, 'iteritems'):
            return to_primitive(dict(value.iteritems()),
                                convert_instances=convert_instances,
                                level=level + 1)
        elif hasattr(value, '__iter__'):
            return to_primitive(list(value),
                                convert_instances=convert_instances,
                                level=level)
        elif convert_instances and hasattr(value, '__dict__'):
            # Likely an instance of something. Watch for cycles.
            # Ignore class member vars.
            instance_dict = to_primitive(value.__dict__,
                                convert_instances=convert_instances,
                                level=level + 1)
            instance_dict['classname'] = value.__class__.__name__
            return instance_dict
        else:
            return value
    except TypeError, e:
        # Class objects are tricky since they may define something like
        # __iter__ defined but it isn't callable as list().
        return unicode(value)