"""Coverage plugin for pytest."""
import os
import warnings

import pytest
import argparse

# from coverage.misc import CoverageException
#
# from . import embed
# from . import engine
# from . import compat

from io import StringIO
from testml.exceptions import TestMlException
from testml import TestMlRunner


class CoverageError(Exception):
    """Indicates that our coverage is too low"""


def validate_report(arg):
    file_choices = ['annotate', 'html', 'xml']
    term_choices = ['term', 'term-missing']
    term_modifier_choices = ['skip-covered']
    all_choices = term_choices + file_choices
    values = arg.split(":", 1)
    report_type = values[0]
    if report_type not in all_choices + ['']:
        msg = 'invalid choice: "{}" (choose from "{}")'.format(arg, all_choices)
        raise argparse.ArgumentTypeError(msg)

    if len(values) == 1:
        return report_type, None

    report_modifier = values[1]
    if report_type in term_choices and report_modifier in term_modifier_choices:
        return report_type, report_modifier

    if report_type not in file_choices:
        msg = 'output specifier not supported for: "{}" (choose from "{}")'.format(arg,
                                                                                   file_choices)
        raise argparse.ArgumentTypeError(msg)

    return values


class SessionWrapper(object):
    def __init__(self, session):
        self._session = session
        if hasattr(session, 'testsfailed'):
            self._attr = 'testsfailed'
        else:
            self._attr = '_testsfailed'

    @property
    def testsfailed(self):
        return getattr(self._session, self._attr)

    @testsfailed.setter
    def testsfailed(self, value):
        setattr(self._session, self._attr, value)


class StoreReport(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        report_type, file = values
        namespace.cov_report[report_type] = file


def pytest_addoption(parser):
    """Add options to control coverage."""

    group = parser.getgroup(
        'ml', 'testml plugin for machine learning models')
    group.addoption('--ml', action='append', default=[], metavar='METRICS',
                    nargs='?', const=True, dest='ml',
                    help='measure coverage for filesystem path '
                         '(multi-allowed)')
    group.addoption('--ml-metrics', action='store', default=[], metavar='METRICS',
                    nargs='?', const=True, dest='ml_metrics',
                    help='measure coverage for filesystem path '
                         '(multi-allowed)')
    group.addoption('--ml-model', action='store', default=None, metavar='MODEL_PATH',
                    nargs='?', const=True, dest='ml_model',
                    help='measure coverage for filesystem path '
                         '(multi-allowed)')
    group.addoption('--ml-loader', action='store', default='infer', metavar='LOADER',
                    nargs='?', const=True, dest='ml_loader',
                    help='measure coverage for filesystem path '
                         '(multi-allowed)')
    group.addoption('--ml-report', action=StoreReport, default={},
                    metavar='type', type=validate_report,
                    help='type of report to generate: console, text, csv, html')

    group.addoption('--ml-data', dest='ml_data', action=StoreReport, default={}, metavar='DATA_PATH',
                    help='type of report to generate: term, term-missing, '
                         'annotate, html, xml (multi-allowed). '
                         'term, term-missing may be followed by ":skip-covered". '
                         'annotate, html and xml may be followed by ":DEST" '
                         'where DEST specifies the output location. '
                         'Use --cov-report= to not generate any output.')

    group.addoption('--ml-config', action='store', default='.mlrc',
                    metavar='path',
                    help='config file for coverage, default: .mlrc')
    group.addoption('--no-ml-cov', dest='no_ml_cov', action='store_true', default=False,
                    help='Disable coverage report completely (useful for debuggers) '
                         'default: False')
    group.addoption('--ml-fail-under', dest='ml_fail_under', action='store', metavar='MIN', type=float,
                    help='Fail if the total coverage is less than MIN.')


@pytest.mark.tryfirst
def pytest_load_initial_conftests(early_config, parser, args):
    plugin = TestMlPlugin(early_config.known_args_namespace, early_config.pluginmanager)
    early_config.pluginmanager.register(plugin, '_test_ml')


class TestMlPlugin(object):
    """Use coverage package to produce code coverage reports.
    Delegates all work to a particular implementation based on whether
    this test process is centralised, a distributed master or a
    distributed slave.
    """

    def __init__(self, options, pluginmanager, start=True):
        """Creates a coverage pytest plugin.
        We read the rc file that coverage uses to get the data file
        name.  This is needed since we give coverage through it's API
        the data file name.
        """

        # Our implementation is unknown at this time.
        print(os.listdir(os.getcwd()))

        self.ml_runner = TestMlRunner(
            config=options.ml_config,
            metrics=options.ml_metrics,
            model=options.ml_model,
            loader=options.ml_loader,
            data=options.ml_data
        )
        self.cov_report = StringIO()
        self.cov_total = None
        self.failed = False
        self._started = False
        self._disabled = False
        self.options = options

        # if getattr(options, 'no_cov', False):
        #     self._disabled = True
        #     return

        # self.options.cov_source = _prepare_cov_source(self.options.cov_source)

        self.start()

        # slave is started in pytest hook

    def start(self, config=None, nodeid=None):

        self.ml_runner.run()
        self._started = True
        if self.options.ml_fail_under:
            self.options.ml_fail_under = 0.5

    def pytest_sessionstart(self, session):
        """At session start determine our implementation and delegate to it."""

        if self.options.no_ml_cov:
            # Coverage can be disabled because it does not cooperate with debuggers well.
            self._disabled = True
            return
        self.start()

    def _should_report(self):
        return not (self.failed and self.options.no_ml_cov)

    def _failed_cov_total(self):
        cov_fail_under = self.options.ml_fail_under
        return cov_fail_under is not None and self.cov_total < cov_fail_under

    # we need to wrap pytest_runtestloop. by the time pytest_sessionfinish
    # runs, it's too late to set testsfailed
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtestloop(self, session):
        yield

        if self._disabled:
            return

        compat_session = SessionWrapper(session)

        self.failed = bool(compat_session.testsfailed)
        if self.ml_runner is not None:
            self.ml_runner.finish()

        if self._should_report():
            try:
                self.cov_total = self.ml_runner.summary(self.cov_report)
            except CoverageException as exc:
                message = 'Failed to generate report: %s\n' % exc
                session.config.pluginmanager.getplugin("terminalreporter").write(
                    'WARNING: %s\n' % message, red=True, bold=True)
                if pytest.__version__ >= '3.8':
                    warnings.warn(pytest.PytestWarning(message))
                else:
                    session.config.warn(code='COV-2', message=message)
                self.cov_total = 0
            assert self.cov_total is not None, 'Test coverage should never be `None`'
            if self._failed_cov_total():
                # make sure we get the EXIT_TESTSFAILED exit code
                compat_session.testsfailed += 1

    def pytest_terminal_summary(self, terminalreporter):
        if self._disabled:
            message = 'Coverage disabled via --no-cov switch!'
            terminalreporter.write('WARNING: %s\n' % message, red=True, bold=True)
            if pytest.__version__ >= '3.8':
                warnings.warn(pytest.PytestWarning(message))
            else:
                terminalreporter.config.warn(code='COV-1', message=message)
            return
        if self.ml_runner is None:
            return

        if self.cov_total is None:
            # we shouldn't report, or report generation failed (error raised above)
            return

        terminalreporter.write('\n' + self.cov_report.getvalue() + '\n')

        if self.options.cov_fail_under is not None and self.options.cov_fail_under > 0:
            if self.cov_total < self.options.cov_fail_under:
                markup = {'red': True, 'bold': True}
                message = (
                        'FAIL Required test coverage of %d%% not '
                        'reached. Total coverage: %.2f%%\n'
                        % (self.options.cov_fail_under, self.cov_total)
                )
            else:
                markup = {'green': True}
                message = (
                        'Required test coverage of %d%% '
                        'reached. Total coverage: %.2f%%\n'
                        % (self.options.cov_fail_under, self.cov_total)
                )
            terminalreporter.write(message, **markup)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        if (item.get_closest_marker('no_cover')
                or 'no_cover' in getattr(item, 'fixturenames', ())):
            self.ml_runner.pause()
            yield
            self.ml_runner.resume()
        else:
            yield


@pytest.fixture
def ml(request):
    """A pytest fixture to provide access to the underlying coverage object."""

    # Check with hasplugin to avoid getplugin exception in older pytest.
    if request.config.pluginmanager.hasplugin('_test_ml'):
        plugin = request.config.pluginmanager.getplugin('_test_ml')
        if plugin.cov_controller:
            return plugin.cov_controller.cov
    return None

