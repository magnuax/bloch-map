# bloch-map

Just a tiny project I threw together for visualizing (qubit) quantum channels as affine transformations of the Bloch sphere.

An example using the map

$$
T\left(\begin{bmatrix} x_1 & x_2 \\\ x_3 & x_4 \end{bmatrix} \right) =
\begin{bmatrix} \frac{1}{3}x_1 + \frac{2}{3}x_4 & \frac{1}{6}x_2 \\\ \frac{1}{6}x_3 & \frac{1}{3}x_4 + \frac{2}{3}x_1 \end{bmatrix}
$$

can be seen below

![alt text](https://github.com/magnuax/bloch-map/blob/main/examples/example.png?raw=true)


Modified version using dash/plotly:

![alt text](https://github.com/magnuax/bloch-map/blob/main/examples/dash.png?raw=true)
