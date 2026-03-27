# Testing and Error Handling Report

This document briefly explains how we test the code and handle errors to keep the system stable.

## Testing Strategy

Since we separated the ROS logic from the core math, we used a **unit-first approach**. We wrote plain Python tests for our algorithms using `pytest`. 

When writing tests, we focused on:
- Making sure the core calculations (like math and geometry) are correct.
- Checking edge cases and boundary conditions.
- Ensuring the code fails predictably when given bad data.

All tests are located in the `test/` directory. You can run them by executing `./scripts/run_tests.sh` from the workspace. It's automated and ready to be plugged into a CI system like GitHub Actions.

## Error Handling

We try to handle errors as cleanly as possible:

- **In the core Python files:** If a function gets bad inputs (like an odd number of coordinates for an x, y list, or a negative speed), it immediately raises a `ValueError`. This fails fast and prevents hidden bugs down the line.
- **In the ROS 2 Nodes:** Before starting up, the nodes check all their ROS parameters. If a user inputs something invalid (like setting the max speed to -5.0), the node will log a friendly warning to the console and automatically fall back to a safe default value.
- **Shutting Down:** The tracking nodes use exception handling to catch external shutdown requests (like pressing Ctrl+C or a launch file tearing down). This ensures they stop cleanly without throwing messy stack traces to the user.
