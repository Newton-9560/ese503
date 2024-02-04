# md
# # Import the package
#
import math
import numpy as np


def new_problem(num):
    print('*' * 25 + 'PROBLEM #{}'.format(num) + '*' * 25)


# md
# # Problem 1
#
new_problem(1)


def get_perimeter(x1, x2, x3, y1, y2, y3):
    """
    Calculates the perimeter of a triangle from its vertex coordinates.
    
    :param x1: X-coordinate of the first vertex.
    :param x2: X-coordinate of the second vertex.
    :param x3: X-coordinate of the third vertex.
    :param y1: Y-coordinate of the first vertex.
    :param y2: Y-coordinate of the second vertex.
    :param y3: Y-coordinate of the third vertex.
    :return: Perimeter of the triangle as a float.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 + ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5 + (
            (x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5


def get_area(x1, x2, x3, y1, y2, y3):
    """
    Calculates the area of a triangle using the vertex coordinates based on the Shoelace formula.
    
    :param x1: X-coordinate of the first vertex.
    :param x2: X-coordinate of the second vertex.
    :param x3: X-coordinate of the third vertex.
    :param y1: Y-coordinate of the first vertex.
    :param y2: Y-coordinate of the second vertex.
    :param y3: Y-coordinate of the third vertex.
    :return: Area of the triangle as a float.
    """
    return 0.5 * abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3))


def is_obtuse(x1, x2, x3, y1, y2, y3):
    """
    Determines if a triangle is obtuse based on its vertex coordinates.
    
    The function calculates the dot products of vectors formed by the triangle's sides
    to determine if any angle is obtuse. An obtuse angle exists if the dot product
    of any two sides' vectors is negative, indicating an angle greater than 90 degrees.
    
    :param x1: X-coordinate of the first vertex.
    :param x2: X-coordinate of the second vertex.
    :param x3: X-coordinate of the third vertex.
    :param y1: Y-coordinate of the first vertex.
    :param y2: Y-coordinate of the second vertex.
    :param y3: Y-coordinate of the third vertex.
    :return: 0 if the triangle is not obtuse, 1 if it is obtuse.
    """
    vector1 = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1)
    vector2 = (x3 - x2) * (x1 - x2) + (y3 - y2) * (y1 - y2)
    vector3 = (x1 - x3) * (x2 - x3) + (y1 - y3) * (y2 - y3)
    if vector1 > 0 and vector2 > 0 and vector3 > 0:
        return 0
    else:
        return 1


#
perimeters = []
areas = []
obtuses = []
for i in range(0, 5000):
    x1, y1, x2, y2, x3, y3 = tuple(np.random.rand(6))
    perimeters.append(get_perimeter(x1, x2, x3, y1, y2, y3))
    areas.append(get_area(x1, x2, x3, y1, y2, y3))
    obtuses.append(is_obtuse(x1, x2, x3, y1, y2, y3))

average_perimeter = np.mean(perimeters)
average_area = np.mean(areas)
probability_obtuse = np.mean(obtuses)

print("The average perimeter of 5000 Monte-Carlo Simulations is {}".format(average_perimeter))
print("The average area of 5000 Monte-Carlo Simulations is {}".format(average_area))
print("The probability that this triangle is obtuse of 5000 Monte-Carlo Simulations is {}".format(probability_obtuse))

# md
# **Result**:
# The average perimeter of 5000 Monte-Carlo Simulations is 1.5659643689857432
# The average area of 5000 Monte-Carlo Simulations is 0.0759041132541655
# The probability that this triangle is obtuse of 5000 Monte-Carlo Simulations is 0.7306
# md
# # Problem 2
#
new_problem(2)


def discriminant(a, b, c):
    """
    Calculates the discriminant of a quadratic equation ax^2 + bx + c = 0.
    
    The discriminant is a mathematical expression (b^2 - 4ac) used to determine the 
    nature of the roots of a quadratic equation. If the discriminant is positive, the 
    equation has two distinct real roots. If it is zero, the equation has exactly one real root.
    If negative, the equation has two complex (imaginary) roots.
    
    :param a: Coefficient of x^2.
    :param b: Coefficient of x.
    :param c: Constant term.
    :return: The value of the discriminant.
    """
    return b * b - 4 * a * c


def have_real_roots(a, b, c):
    """
    Determines if a quadratic equation has real roots based on its coefficients.
    
    This function uses the discriminant (b^2 - 4ac) of the quadratic equation ax^2 + bx + c = 0
    to check for the presence of real roots. A quadratic equation has real roots if its discriminant
    is greater than or equal to zero.
    
    :param a: Coefficient of x^2.
    :param b: Coefficient of x.
    :param c: Constant term.
    :return: 1 if the equation has real roots, 0 otherwise.
    """
    if discriminant(a, b, c) >= 0:
        return 1
    else:
        return 0


def solve_equation(a, b, c):
    """
    Solves a quadratic equation ax^2 + bx + c = 0 and returns its roots.
    
    This function calculates the roots using the quadratic formula: (-b ± sqrt(b^2 - 4ac)) / (2a).
    It assumes that the discriminant (D = b^2 - 4ac) is non-negative, which means the equation
    has real roots. If D is negative, math.sqrt(D) will raise a ValueError due to taking the square root
    of a negative number.
    
    :param a: Coefficient of x^2, must be non-zero.
    :param b: Coefficient of x.
    :param c: Constant term.
    :return: A tuple containing the two roots of the quadratic equation. If the discriminant is zero,
             the two roots are identical.
    """
    D = discriminant(a, b, c)
    root1 = (-b + math.sqrt(D)) / (2 * a)
    root2 = (-b - math.sqrt(D)) / (2 * a)
    return root1, root2


#
have_real_roots_list = []
for i in range(5000):
    a, b, c = tuple(np.random.rand(3))
    have_real_roots_list.append(have_real_roots(a, b, c))

roots1 = []
roots2 = []
while True:
    a, b, c = tuple(np.random.rand(3))
    if discriminant(a, b, c) >= 0:
        root1, root2 = solve_equation(a, b, c)
        roots1.append(root1)
        roots2.append(root2)
        if len(roots1) > 5000:
            break

print("The probability that both roots of this equation are real of 5000 Monte-Carlo Simulations is {}".format(
    np.mean(have_real_roots_list)))
print("The root1 of this equation of 5000 Monte-Carlo Simulations is {}".format(np.mean(roots1)))
print("The root2 of this equation of 5000 Monte-Carlo Simulations is {}".format(np.mean(roots2)))

# md
# **Result:**
# The probability that both roots of this equation are real of 5000 Monte-Carlo Simulations is 0.2546
# The root1 of this equation of 5000 Monte-Carlo Simulations is -0.5767142878400107
# The root2 of this equation of 5000 Monte-Carlo Simulations is -14.32722766903135
# md
# # Problem 3
new_problem(3)


def distance_point_to_line_segment(A, B, C):
    """
    Calculates the shortest distance from a point to a line segment.
    
    Given a line segment defined by points A and B, and a point C, this function calculates 
    the shortest distance from C to the line segment AB. It does so by projecting point C onto 
    the line defined by A and B, determining if the projected point lies within the segment, 
    and calculating the distance from C to the closest point on AB.
    
    :param A: The starting point of the line segment as a NumPy array.
    :param B: The ending point of the line segment as a NumPy array.
    :param C: The point from which the distance is measured as a NumPy array.
    :return: The shortest distance from point C to the line segment AB as a float.
    """
    AB = B - A
    AC = C - A
    t = np.dot(AC, AB) / np.dot(AB, AB)

    if t < 0.0:
        nearest_point = A
    elif t > 1.0:
        nearest_point = B
    else:
        nearest_point = A + t * AB

    return np.linalg.norm(C - nearest_point)


def intersect(A, B, C, R):
    """
    Determines if a circle intersects with a line segment.
    
    This function checks if the shortest distance from the center of the circle (point C) to 
    the line segment AB is less than or equal to the radius of the circle (R). If so, it indicates 
    that the circle and the line segment intersect.
    
    :param A: The starting point of the line segment as a NumPy array.
    :param B: The ending point of the line segment as a NumPy array.
    :param C: The center of the circle as a NumPy array.
    :param R: The radius of the circle as a float.
    :return: 1 if the circle and the line segment intersect, 0 otherwise.
    """
    if distance_point_to_line_segment(A, B, C) <= R:
        return 1
    else:
        return 0


#
is_intersect = []
for i in range(5000):
    end1 = np.random.rand(2)
    end2 = np.random.rand(2)
    center = np.random.rand(2)
    radius = np.random.rand()
    is_intersect.append(intersect(end1, end2, center, radius))

print('The probability that the line intersects the circle of 5000 Monte-Carlo Simulations is {}'.format(
    np.mean(is_intersect)))
# md
# **Result:**
# The probability that the line intersects the circle of 5000 Monte-Carlo Simulations is 0.6378
# md
# # Problem 4
new_problem(4)
tuition = 8400
dormitory = 5400

state_scholarship = 3000
parental_contributions = 4000


def random_number(start, end):
    """
    Generates a random floating-point number within the specified range.
    
    :param start: The lower bound of the range (inclusive).
    :param end: The upper bound of the range (exclusive).
    :return: A random floating-point number between start and end.
    """
    return np.random.rand() * (end - start) + start


def loan_to_range(loan):
    """
    Categorizes a loan amount into predefined ranges.
    
    The function categorizes the loan amount based on its value:
    - Returns -1 for loans >= 5500.
    - Returns 1 for loans in the range [0, 1500).
    - For loans between 1500 and 5500, it categorizes them into ranges of 500
      (e.g., [1500, 2000), [2000, 2500), ..., [5000, 5500)) and returns the
      range number (1 to 8).
    - Returns None if the loan amount does not fit into any of the categories,
      though this condition may not be reached with the current logic.
    
    :param loan: The loan amount as a float or integer.
    :return: The category or range number of the loan, -1, a number from 1 to 8, or None.
    """
    if loan >= 5500:
        return -1
    if 0 <= loan < 1500:
        return 1

    for i in range(1, 9):
        if 500 * i + 1000 <= loan < 500 * i + 1500:
            return i

    return None


#
distribution = np.zeros(10)

for i in range(1000):
    meals = random_number(start=900, end=1350)
    entertainment = random_number(start=600, end=1200)
    transportation = random_number(start=200, end=600)
    books = random_number(start=400, end=800)

    waiting_tables = random_number(start=3000, end=5000)
    library_aide = random_number(start=2000, end=3000)

    loan = -(
            waiting_tables + library_aide + state_scholarship + parental_contributions - tuition - dormitory - meals - entertainment - transportation - books)

    r = loan_to_range(loan)
    if r is not None:
        distribution[r] += 1

print("The probability distribution for the size of the loan of 1000 simulations is {}".format(
    distribution / np.sum(distribution)))
# md
# **Result:**
# The probability distribution for the size of the loan of 1000 simulations is [0.    0.027 0.113 0.184 0.264 0.229 0.13  0.05  0.003 0.   ]
# md
# # Problem 5
new_problem(5)
bus_arrive = np.array([0, 20, 40, 60, 80, 100])


def wait_time(passenger_t):
    """
    Calculates the wait time for the next bus based on the passenger's arrival time.
    
    Iterates through a list of bus arrival times (`bus_arrive`) to find the first bus
    that arrives after the passenger. The function then calculates and returns the wait
    time by subtracting the passenger's arrival time from the bus's arrival time.
    
    :param passenger_t: The arrival time of the passenger.
    :return: The wait time for the next bus. If no bus arrives after the passenger's arrival time, the function returns `None`.
    """
    for bus_time in bus_arrive:
        if passenger_t <= bus_time:
            return bus_time - passenger_t


#
wait_times = []
more_than5 = []
for i in range(1000):
    passenger_t = random_number(10, 90)
    wait = wait_time(passenger_t)
    wait_times.append(wait)
    if wait > 5:
        more_than5.append(1)
    else:
        more_than5.append(0)

print(
    "The probability that the passenger will have to wait more than 5 minutes for a bus of 1000 simulations is {}".format(
        np.mean(more_than5)))

print("The average time that the passenger will wait for a bus of 1000 simulations is {}".format(np.mean(wait_times)))
# md
# **Result:**
# The probability that the passenger will have to wait more than 5 minutes for a bus of 1000 simulations is 0.749
# The average time that the passenger will wait for a bus of 1000 simulations is 10.0911832564719
# 
# md
# # Problem 6
new_problem(6)
intersects = []

for i in range(5000):
    line1_endA = np.random.rand(2)
    line1_endB = np.random.rand(2)
    line2_endA = np.random.rand(2)
    line2_endB = np.random.rand(2)
    a = np.cross(line1_endA - line1_endB, line1_endA - line2_endA) * np.cross(line1_endA - line1_endB,
                                                                              line1_endA - line2_endB)
    b = np.cross(line2_endA - line2_endB, line2_endA - line1_endA) * np.cross(line2_endA - line2_endB,
                                                                              line2_endA - line1_endB)
    if a <= 0 and b <= 0:
        intersects.append(1)
    else:
        intersects.append(0)

print("The probability that the two line-segments intersect of 5000 simulations is {}".format(np.mean(intersects)))
# md
# **Result:**
# The probability that the two line-segments intersect of 5000 simulations is 0.238
# md
# # Problem 7 
new_problem(7)
intersects = []
for i in range(5000):
    A = np.random.rand(2)
    B = np.random.rand(2)
    R1 = np.random.uniform(0, min(A[0], 1 - A[0], A[1], 1 - A[1]))
    R2 = np.random.uniform(0, min(B[0], 1 - B[0], B[1], 1 - B[1]))
    if np.linalg.norm(A - B) < R1 + R2:
        intersects.append(1)
    else:
        intersects.append(0)

print("The probability that the two disks intersect of 5000 simulations is {}".format(np.mean(intersects)))
# md
# **Result:**
# The probability that the two disks intersect of 5000 simulations is 0.1244
