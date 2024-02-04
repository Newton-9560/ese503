# ESE503

## hw1: Monte-Carlo Simulations

### Description of the Provided Code

The provided code consists of a series of problem statements, each defined by a specific mathematical or simulation task. These tasks range from geometric calculations, such as determining the perimeter and area of a triangle, to more complex Monte Carlo simulations for estimating probabilities in various scenarios. Here's a breakdown:

### Import the package
The code begins by importing necessary Python libraries, `math` for mathematical functions and `numpy` for numerical operations.

### Helper Function
- `new_problem(num)`: Prints a header to denote the start of a new problem, enhancing readability in the output.

#### Problem 1: Triangle Geometry
- Defines functions to calculate the perimeter and area of a triangle (`get_perimeter`, `get_area`) and to determine if a triangle is obtuse (`is_obtuse`).
- Conducts 5000 simulations to generate random triangles and computes their average perimeter, average area, and the probability of being obtuse.

#### Problem 2: Quadratic Equations
- Introduces functions for calculating the discriminant (`discriminant`), checking if real roots exist (`have_real_roots`), and solving quadratic equations (`solve_equation`).
- Uses Monte Carlo simulations to estimate the probability of a randomly generated quadratic equation having real roots and calculates average values of the roots.

#### Problem 3: Geometry and Intersection
- Includes a function to calculate the distance from a point to a line segment (`distance_point_to_line_segment`) and another to determine if a circle intersects with a line segment (`intersect`).
- Performs simulations to estimate the probability of intersection between randomly generated circles and line segments.

#### Problem 4: Financial Simulation
- Simulates a student's financial needs over a semester, considering various expenses and incomes.
- Estimates the distribution of loan amounts required by the student using Monte Carlo methods.

#### Problem 5: Bus Waiting Time
- Simulates a passenger waiting for a bus, with predefined bus arrival times.
- Calculates and reports the probability of waiting more than 5 minutes and the average wait time based on simulations.

#### Problem 6: Line Segment Intersection
- Simulates the intersection of two randomly placed line segments using vector cross products.
- Estimates the probability of intersection through simulations.

#### Problem 7: Disk Intersection
- Similar to Problem 6 but focuses on the intersection of two randomly placed disks within a unit square.
- Uses Monte Carlo simulations to estimate the probability of disk intersections.

Each problem is self-contained, with its own set of simulations and print statements to display results, showcasing applications of mathematical concepts and simulation techniques in Python.
