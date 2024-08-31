/* Loss functions in linear regression
 
                    GNU AFFERO GENERAL PUBLIC LICENSE
                       Version 3, 19 November 2007

    Copyright (C) 2024 Debajyoti Debnath

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include "matrix.h"
#include "losses.h"

// l2 loss = ||y_true - y_pred||^2/N
// where y_true is ground truth,
// y_pred is predicted values and N is the 
// number of observations.
double l2_loss(Matrix* y_true, Matrix* y_pred)
{
    Matrix diff;
    double loss;
    
    // diff = y_true - y_pred
    diff = mat_copy(y_true);
    mat_sub(&diff, y_pred);

    // loss = ||diff||^2
    loss = mat_norm(&diff);
    loss = loss*loss;

    mat_destroy(&diff);

    return loss / y_pred->nrows;
}
 
// Gradient of the L2 loss with respect to theta (coefficients)
// given by gradient = X.T (X theta - y)
Matrix l2_gradient(Matrix* x, Matrix* y, Matrix* theta)
{
    Matrix grad, x_theta_prod;
    
    // Calculate gradient
    x_theta_prod = mat_mul(x, false, theta, false);
    mat_sub(&x_theta_prod, y);
    grad = mat_mul(x, true, &x_theta_prod, false);

    mat_destroy(&x_theta_prod);
 
    return grad;
}
