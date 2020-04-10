from torch.utils.cpp_extension import load
import sys
linepoolalign = load(
    'linepoolalign',
    sources=['extension/line_alignpooling/LinePoolAlign.cpp',
             'extension/line_alignpooling/LinePoolAlign_cuda.cu'],
    verbose=True,
    extra_include_paths=['/usr/include/python{}.{}/'.format(sys.version_info.major, sys.version_info.minor)])
help(linepoolalign)
