!   QSW_MPI -  A package for parallel Quantum Stochastic Walk simulation.
!   Copyright (C) 2019 Edric Matwiejew
!
!   This program is free software: you can redistribute it and/or modify
!   it under the terms of the GNU General Public License as published by
!   the Free Software Foundation, either version 3 of the License, or
!   (at your option) any later version.
!
!   This program is distributed in the hope that it will be useful,
!   but WITHOUT ANY WARRANTY; without even the implied warranty of
!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!   GNU General Public License for more details.
!
!   You should have received a copy of the GNU General Public License
!   along with this program.  If not, see <https://www.gnu.org/licenses/>.

!> @brief Imports numerical precisions.
!> @details Defines single precision (sp), double precision (dp) and quadrupule presision (qp) kind types using the 
!> instrinsic *iso_fortran_env* module types real32, real64 and real128 respectively.

module ISO_Precisions

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64, qp => real128

end module ISO_Precisions


