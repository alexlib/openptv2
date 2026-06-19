try:
	from traits.etsconfig.etsconfig import ETSConfig

	ETSConfig.toolkit = "qt"
except ModuleNotFoundError:
	# Traits is an optional dependency for headless/non-GUI usage.
	# GUI entrypoints will still require traits/traitsui to be installed.
	pass


def __getattr__(name):
	if name == "__version__":
		from .__version__ import __version__
		return __version__
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

