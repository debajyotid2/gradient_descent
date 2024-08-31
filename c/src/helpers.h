/* Helper functions

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

#ifndef _HELPERS_H_
#define _HELPERS_H_

#include <stdlib.h>
#include <stdbool.h>
#include <matrix.h>

void make_regression_dataset(Matrix* x, Matrix* y,
                             double bias, double noise_intensity,
                             unsigned int seed);
void split_into_train_test(Matrix* x, Matrix* y,
                           Matrix* x_train, Matrix* y_train,
                           Matrix* x_test, Matrix* y_test,
                           unsigned int seed);

#endif // _HELPERS_H_
