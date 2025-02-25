//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_swap_operators.h
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
// Maintainer: DePrince group
//
// This file is part of the pdaggerq package.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#ifndef PQ_SWAP_OPERATORS_H
#define PQ_SWAP_OPERATORS_H

#include<string>
#include<vector>

#include "pq_string.h"

namespace pdaggerq {

/**
 *
 * swap two operators in a string to bring that string toward normal order with respect to the fermi vacuum
 *
 * @param in: the input string
 * @param ordered: a list of strings to which the new strings will be added after applying appropriate rules for the swap
 */
bool swap_operators_fermi_vacuum(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &ordered);

/**
 *
 * swap two operators in a string to bring that string toward normal order with respect to the true vacuum
 *
 * @param in: the input string
 * @param ordered: a list of strings to which the new strings will be added after applying appropriate rules for the swap
 */
bool swap_operators_true_vacuum(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &ordered);

}

#endif 
