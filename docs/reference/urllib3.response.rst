Response and Decoders
=====================

Response
--------

.. autoclass:: urllib3.response.BaseHTTPResponse
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: urllib3.response.HTTPResponse
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: json

    .. autoattribute:: auto_close
    .. autoattribute:: status
    .. autoattribute:: headers

Consumption state
-----------------

When a response is created with ``track_consumption_state=True`` (or the
same option is passed through ``urlopen()``/``request()``),
:attr:`urllib3.response.BaseHTTPResponse.consumption_state` returns a live
record of how the response body has been consumed and who owns its
connection. Querying the record never performs a network read.

.. autoclass:: urllib3.response.ResponseConsumptionState
    :members:
    :undoc-members:

.. autoclass:: urllib3.response.ConnectionOwnership
    :members:
    :undoc-members:

Decoders
--------

Decoder classes are used for transforming compressed HTTP bodies
using the ``Content-Encoding`` into their uncompressed binary
representation.

.. autoclass:: urllib3.response.BrotliDecoder
.. autoclass:: urllib3.response.DeflateDecoder
.. autoclass:: urllib3.response.GzipDecoder
.. autoclass:: urllib3.response.ZstdDecoder
.. autoclass:: urllib3.response.MultiDecoder
