# AI Agent Guide for Rubik's Cube Project

## Project Overview

This is a **Rubik's Cube Solver and Analytics Engine** - a comprehensive Python application built with Streamlit that provides tools for solving, analyzing, and working with n×n Rubik's cubes.

**Key Features:**

- 🏷️ **Autotagger**: Automatic tagging of cube states and move sequences
- 🧩 **Solver**: Bidirectional search solver with configurable algorithms
- 🎨 **Pattern**: Pattern analysis and custom pattern creation
- 📚 **Docs**: Documentation and reference materials

## Project Structure

```bash
rubiks_cube/
├── app.py               # Main Streamlit application entry point
├── pages.py             # UI pages (autotagger, solver, pattern, docs)
├── configuration/       # App configuration and settings
│   ├── __init__.py      # Global constants (CUBE_SIZE, METRIC, APP_MODE)
│   ├── enumeration.py   # Enums for metrics, pieces, steps, etc.
│   ├── logging.py       # Logging configuration with file rotation
│   ├── paths.py         # Path constants for data/logs/resources
│   └── types.py         # Type definitions for cube states/patterns
├── move/                # Move representation and generation
│   ├── algorithm.py     # Move algorithms and collections
│   ├── generator.py     # Move generators for different cube types
│   ├── sequence.py      # Move sequences and cleanup operations
│   ├── metrics.py       # Move counting metrics (HTM, QTM, etc.)
│   └── utils.py         # Move utilities and transformations
├── representation/      # Cube state representation
│   ├── mask.py          # Cube masks for pattern matching
│   ├── pattern.py       # Cube patterns and pattern operations
│   ├── permutation.py   # Cube permutations and transformations
│   ├── symmetries.py    # Cube symmetries and rotations
│   └── utils.py         # State utilities and conversions
├── solver/              # Cube solving algorithms
│   ├── __init__.py      # Main solving interface
│   ├── actions.py       # Action space generation
│   ├── bidirectional_solver.py  # Bidirectional search implementation
│   ├── heuristics.py    # Solving heuristics
│   ├── optimizers.py    # Solution optimization
│   ├── search.py        # Search algorithms
│   └── solver_abc.py    # Solver abstract base class
├── tag/                 # Cube state tagging system
│   ├── cubex.py         # Cubex collections for pattern matching
│   └── tracing.py       # Tag tracing and analysis
├── formatting/          # Text formatting and parsing
│   ├── decorator.py     # Move decoration utilities
│   ├── regex.py         # Regular expressions for move parsing
│   └── string.py        # String formatting utilities
├── parsing/             # Input parsing
│   └── __init__.py      # Parse scrambles and move sequences
├── graphics/            # Visualization and graphics
│   ├── horizontal.py    # 2D cube visualization with matplotlib
│   └── svg_icon.py      # SVG icon generation
├── attempt/             # Solve attempt management
└── data/                # Data directory (auto-created)
    ├── logs/            # Application logs
    └── resources/       # Static resources
```

## Technology Stack

- **Framework**: Streamlit (web interface)
- **Package Manager**: uv (modern Python package manager)
- **Language**: Python 3.13+ with strict typing
- **Visualization**: Matplotlib, SVG generation
- **Data**: NumPy for numerical operations
- **Logging**: JSON structured logging with rotation
- **Quality**: Black, Ruff, MyPy, pre-commit hooks

## Development Workflow

### Setup

```bash
# Install dependencies
uv sync

# Run application
uv run streamlit run rubiks_cube/app.py

# Run tests
uv sync --extra dev
uv run pytest

# Code quality
uv run pre-commit run --all-files
```

### Logging

- Logs to `rubiks_cube/data/logs/rubiks_cube.log`
- JSON format with rotation (5MB files, 3 backups)
- Auto-creates log directory if missing

## Common Development Tasks

### Adding New Move Types

1. Update `move/generator.py` with new move patterns
2. Add parsing support in `formatting/regex.py`
3. Update move validation in `formatting/string.py`

### Adding New Cube States

1. Define types in `configuration/types.py`
2. Add conversion utilities in `representation/utils.py`
3. Update visualization in `graphics/`

### Adding New Solvers

1. Inherit from `solver/solver_abc.py`
2. Implement required abstract methods
3. Register in `solver/__init__.py`

### Adding New UI Features

1. Add page function to `pages.py`
2. Update router in `app.py`
3. Add route handling

## Code Quality Standards

- **Type Safety**: Full MyPy strict mode compliance
- **Formatting**: Black with 100-character line length
- **Linting**: Ruff with extensive rule set
- **Testing**: Pytest with comprehensive test coverage
- **Documentation**: Google-style docstrings required

## Debugging Tips

- Check `rubiks_cube/data/logs/rubiks_cube.log` for detailed logs
- Use `APP_MODE="development"` for debug logging
- Streamlit provides built-in error handling and stack traces
- Test individual components with direct Python execution

## Integration Points

- **Streamlit**: Web interface framework
- **NumPy**: Numerical operations on cube states
- **Matplotlib**: 2D cube visualization
- **Typer**: CLI tools for graphics generation
- **Extra Streamlit Components**: Enhanced UI widgets

## Contributing Guidelines

1. Follow existing code structure and naming conventions
2. Add comprehensive type hints to all functions
3. Include docstrings for public APIs
4. Add tests for new functionality
5. Run pre-commit hooks before submitting
6. Update this guide when adding major features
