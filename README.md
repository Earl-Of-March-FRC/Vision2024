
# EOM Vision Subteam Repository

This repository contains the codebase for the Vision Subteam of FRC Lioneers. 

## Introduction

This repository hosts the codebase for our vision processing system. It includes object detection algorithms, network communication scripts, and other related components.

## Dependencies

- [Python](https://www.python.org/) (version 3.7 or later)
- [OpenCV](https://opencv.org/) (version 4.0 or later)
- [numpy](https://numpy.org/) (version 1.20 or later)
- [ntcore](https://github.com/wpilibsuite/ntcore) (version 4.0 or later)
- [ultralytics](https://github.com/ultralytics/yolov5) (version 6.0 or later)

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Earl-Of-March-FRC/Vision2024.git
```

2. Install the dependencies:

3. Run the main script:

```bash
python main.py
```

## Contributing

We welcome contributions from team members. If you have any improvements, bug fixes, or new features, please follow these steps:

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -am 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## Branching Strategy

We follow a simple branching strategy to manage our codebase efficiently:

- **main**: This branch contains stable, production-ready code. It should always be deployable.
- **development**: All feature development and integration work should be done in this branch. It's where new features are developed and tested.
- **feature branches**: Each feature or task should have its own branch, branched off from the development branch. Once complete, it will be merged back into development.
- **referemce**: Where all reference dependencies for Vision team is located.

## Code Style

Please ensure that your code follows the PEP 8 style guide for Python.
