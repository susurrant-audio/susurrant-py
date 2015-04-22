#!/usr/bin/env python
import subprocess
import codecs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_app(executable, ordered_opts=[],
            kw_opts={},
            progress_file=None):
    process_args = [executable] + ordered_opts

    for (k, v) in kw_opts.iteritems():
        process_args.append(unicode(k))
        if v is not None:
            process_args.append(unicode(v))

    logger.info('Cmd: ' + ' '.join(process_args))
    if progress_file is not None:
        with codecs.open(progress_file, 'w', encoding='utf-8') as prog:
            subprocess.call(process_args,
                            stdout=prog,
                            stderr=prog)
    else:
        subprocess.call(process_args)
