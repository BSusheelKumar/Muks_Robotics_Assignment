.PHONY: build test test-report clean

build:
	source /opt/ros/humble/setup.bash && colcon build --packages-select muks_robotics_assignment

test:
	python3 -m pytest src/muks_robotics_assignment/test

test-report:
	src/muks_robotics_assignment/scripts/run_tests.sh

clean:
	rm -rf build install log
