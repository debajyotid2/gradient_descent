/* Miscellaneous statistical functions
 
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

#ifndef _STATS_H_
#define _STATS_H_

#include <stdlib.h>
#include <matrix.h>

Matrix stats_mean(Matrix* mat, unsigned int dimension);
double stats_mae(Matrix* y_true, Matrix* y_pred);
double stats_r2(Matrix* y_true, Matrix* y_pred);

#endif // _STATS_H_
