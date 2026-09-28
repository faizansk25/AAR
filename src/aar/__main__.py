"""Enable ``python -m aar`` as an alternative to the ``aar`` console script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

