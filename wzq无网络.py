# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import math
import time
import pickle
from collections import defaultdict, namedtuple
from typing import Tuple, List, Dict, Optional, Any, Sequence
import traceback
import sys
import datetime
import multiprocessing
import cProfile
import pstats
import io
# import builtins # Removed

# --- Numba Setup ---
try:
    import numba
    from numba import njit, prange, types
    NUMBA_AVAILABLE = True
except ImportError:
    print("错误：未找到 Numba。此代码需要 Numba 才能高效运行。")
    print("请安装 Numba: pip install numba")
    sys.exit(1)

# --- Pygame Setup ---
try:
    import pygame # type: ignore
    if pygame.get_init(): pygame.quit()
    PYGAME_AVAILABLE = True
except ImportError: PYGAME_AVAILABLE = False
except pygame.error: PYGAME_AVAILABLE = True

# --- Game Constants ---
BOARD_SIZE = 15
WIN_TARGET = 5
if WIN_TARGET > BOARD_SIZE: print(f"警告: WIN_TARGET ({WIN_TARGET}) 大于 BOARD_SIZE ({BOARD_SIZE})!")

# --- Alpha-Beta Parameters ---
DEFAULT_SEARCH_DEPTH = 4
DEFAULT_TIME_LIMIT_SECONDS = 10
INF = float('inf')
TT_SIZE_LIMIT = 1000000
TT_ENTRY_FLAGS = {'EXACT': 0, 'LOWER': 1, 'UPPER': 2}
MAX_KILLER_MOVES = 2
QUIESCENCE_SEARCH_DEPTH = 2
QSEARCH_TIME_CHECK_INTERVAL = 64

# --- Evaluation Weights (Further Refined for Strategy) ---
EVAL_WEIGHTS = {
    'WIN':              100000000,
    'BLOCK_WIN':        95000000,
    'LIVE_FOUR':        6000000,   # AI 自己的活四 (略微提高)
    'BLOCK_LIVE_FOUR':  96000000,  # 阻止对方活四 (极高)
    'DEAD_FOUR':        80000,     # AI 自己的冲四
    'BLOCK_DEAD_FOUR':  500000,    # 阻止对方冲四 (高于活三)
    'LIVE_THREE':       60000,     # AI 自己的活三
    'BLOCK_LIVE_THREE': 85000,     # 阻止对方活三 (明显高于自己活三, 也高于自己冲四)
    'DEAD_THREE':       500,
    'BLOCK_DEAD_THREE': 550,       # 略高于对方死三
    'LIVE_TWO':         500,
    'BLOCK_LIVE_TWO':   550,       # 略高于对方活二
    'DEAD_TWO':         5,
    'BLOCK_DEAD_TWO':   4,
    'POS_WEIGHT_MAX':   10,
    'DEFENSE_WEIGHT':   1.15,      # 保持稍高的防御倾向
    'HEURISTIC_POS_WEIGHT_SCALE': 5.0,
}

# --- Player Representation ---
EMPTY = 0; PLAYER_X = 1; PLAYER_O = -1

# --- Numba Optimized Helpers (Types) ---
intp = numba.intp; int8 = numba.int8; float64 = numba.float64; boolean = numba.boolean
uint64 = numba.uint64
tuple_i8_i8 = types.Tuple((intp, intp))

# --- Regenerate WEIGHTS_TUPLE with updated EVAL_WEIGHTS ---
WEIGHTS_TUPLE_LIST = [
    float(EVAL_WEIGHTS['WIN']), float(EVAL_WEIGHTS['BLOCK_WIN']),
    float(EVAL_WEIGHTS['LIVE_FOUR']), float(EVAL_WEIGHTS['BLOCK_LIVE_FOUR']),
    float(EVAL_WEIGHTS['DEAD_FOUR']), float(EVAL_WEIGHTS['BLOCK_DEAD_FOUR']),
    float(EVAL_WEIGHTS['LIVE_THREE']), float(EVAL_WEIGHTS['BLOCK_LIVE_THREE']),
    float(EVAL_WEIGHTS['DEAD_THREE']), float(EVAL_WEIGHTS['BLOCK_DEAD_THREE']),
    float(EVAL_WEIGHTS['LIVE_TWO']), float(EVAL_WEIGHTS['BLOCK_LIVE_TWO']),
    float(EVAL_WEIGHTS['DEAD_TWO']), float(EVAL_WEIGHTS['BLOCK_DEAD_TWO']),
    float(EVAL_WEIGHTS['DEFENSE_WEIGHT'])
]
weights_tuple_type = types.UniTuple(numba.float64, len(WEIGHTS_TUPLE_LIST))
WEIGHTS_TUPLE = tuple(WEIGHTS_TUPLE_LIST) # Ensure this tuple uses the final weights

killer_tuple_type = types.UniTuple(numba.intp, MAX_KILLER_MOVES)

# --- Zobrist Hashing Initialization ---
ZOBRIST_TABLE = np.random.randint(1, np.iinfo(np.uint64).max,
                                  size=(BOARD_SIZE * BOARD_SIZE, 3), dtype=np.uint64)
ZOBRIST_SIDE_TO_MOVE = np.random.randint(1, np.iinfo(np.uint64).max, dtype=np.uint64)

def get_zobrist_piece_index(piece):
    if piece == PLAYER_X: return 1
    if piece == PLAYER_O: return 2
    return 0

# --- Precompute Positional Weights ---
positional_weights = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
center = BOARD_SIZE // 2
for r in range(BOARD_SIZE):
    for c in range(BOARD_SIZE):
        dist = max(abs(r - center), abs(c - center))
        weight = max(0, EVAL_WEIGHTS['POS_WEIGHT_MAX'] * (1 - dist / (center + 1e-6)))
        positional_weights[r, c] = weight
positional_weights_flat = positional_weights.flatten()

# --- Numba Functions ---
analyze_sig = types.UniTuple(intp, 2)(int8[:], intp, intp, intp, intp, intp, intp, intp)
@njit(analyze_sig, cache=True, fastmath=True, nogil=True)
def _analyze_line_segment_numba(board_flat, board_size, r, c, dr, dc, player, win_target):
    idx_start = r * board_size + c
    if idx_start < 0 or idx_start >= len(board_flat) or board_flat[idx_start] != player: return 0, 0
    pr, pc = r - dr, c - dc
    open_before = False
    if 0 <= pr < board_size and 0 <= pc < board_size:
        prev_idx = pr * board_size + pc
        if prev_idx >= 0 and prev_idx < len(board_flat):
            prev_piece = board_flat[prev_idx]
            if prev_piece == player: return 0, 0
            if prev_piece == EMPTY: open_before = True
        else: open_before = False
    else: open_before = False
    consecutive = 1; open_after = False
    for i in range(1, win_target + 1):
        nr, nc = r + i * dr, c + i * dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            idx = nr * board_size + nc
            if idx >= 0 and idx < len(board_flat):
                piece = board_flat[idx]
                if piece == player:
                    if consecutive < win_target: consecutive += 1
                elif piece == EMPTY: open_after = True; break
                else: open_after = False; break
            else: open_after = False; break
        else: open_after = False; break
    open_ends = int(open_before) + int(open_after)
    return consecutive, open_ends

@njit(cache=True, fastmath=True, nogil=True)
def static_evaluation_numba_core(board_flat_int8: np.ndarray, board_size: int, win_target: int, player: int,
                                 pos_weights_flat: np.ndarray,
                                 w_win: float, w_block_win: float,
                                 w_lf: float, w_blf: float, w_df: float, w_bdf: float,
                                 w_lt: float, w_blt: float, w_dt: float, w_bdt: float,
                                 w_l2: float, w_bl2: float, w_d2: float, w_bd2: float,
                                 defense_weight: float) -> float:
    opponent = -player; my_score = 0.0; opponent_score = 0.0
    directions = np.array([(0, 1), (1, 0), (1, 1), (1, -1)], dtype=np.int8)
    my_patterns = np.zeros(6, dtype=np.intp); opp_patterns = np.zeros(6, dtype=np.intp)
    my_has_win = False; opp_has_win = False
    occupied_indices = np.where(board_flat_int8 != EMPTY)[0]
    for idx_base in occupied_indices:
        r = idx_base // board_size; c = idx_base % board_size
        piece = board_flat_int8[idx_base]
        current_player_patterns = my_patterns if piece == player else opp_patterns
        is_my_piece = (piece == player)
        for d_idx in range(len(directions)):
            dr, dc = directions[d_idx]
            pr, pc = r - dr, c - dc
            is_start_of_line = True
            if 0 <= pr < board_size and 0 <= pc < board_size:
                 if board_flat_int8[pr * board_size + pc] == piece:
                     is_start_of_line = False
            if is_start_of_line:
                consecutive, open_ends = _analyze_line_segment_numba(board_flat_int8, board_size, r, c, dr, dc, piece, win_target)
                if consecutive >= win_target:
                    if is_my_piece: my_has_win = True
                    else: opp_has_win = True
                    if my_has_win or opp_has_win: break
                elif consecutive == win_target - 1:
                    if open_ends == 2: current_player_patterns[0] += 1
                    elif open_ends == 1: current_player_patterns[1] += 1
                elif consecutive == win_target - 2:
                    if open_ends == 2: current_player_patterns[2] += 1
                    elif open_ends == 1: current_player_patterns[3] += 1
                elif consecutive == win_target - 3 and win_target > 3:
                    if open_ends == 2: current_player_patterns[4] += 1
                    elif open_ends == 1: current_player_patterns[5] += 1
        if my_has_win or opp_has_win: break
    if my_has_win: return w_win
    if opp_has_win: return -w_block_win
    my_score += my_patterns[0] * w_lf + my_patterns[1] * w_df
    my_score += my_patterns[2] * w_lt + my_patterns[3] * w_dt
    my_score += my_patterns[4] * w_l2 + my_patterns[5] * w_d2
    my_score += opp_patterns[0] * w_blf + opp_patterns[1] * w_bdf
    my_score += opp_patterns[2] * w_blt + opp_patterns[3] * w_bdt
    my_score += opp_patterns[4] * w_bl2 + opp_patterns[5] * w_bd2
    opponent_score += opp_patterns[0] * w_lf + opp_patterns[1] * w_df
    opponent_score += opp_patterns[2] * w_lt + opp_patterns[3] * w_dt
    opponent_score += opp_patterns[4] * w_l2 + opp_patterns[5] * w_d2
    opponent_score += my_patterns[0] * w_blf + my_patterns[1] * w_bdf
    opponent_score += my_patterns[2] * w_blt + my_patterns[3] * w_bdt
    opponent_score += my_patterns[4] * w_bl2 + my_patterns[5] * w_bd2
    pos_score_diff = 0.0
    for idx_base in occupied_indices:
        piece = board_flat_int8[idx_base]
        if piece == player: pos_score_diff += pos_weights_flat[idx_base]
        elif piece == opponent: pos_score_diff -= pos_weights_flat[idx_base]
    final_score = (my_score + pos_score_diff) - opponent_score * defense_weight
    return final_score

@njit(boolean(int8[:], intp, intp, intp, intp, intp), cache=True, fastmath=True, nogil=True)
def _check_win_around_pos_numba(board_flat, board_size, win_target, r, c, player):
    if not (0 <= r < board_size and 0 <= c < board_size): return False
    directions = np.array([(0, 1), (1, 0), (1, 1), (1, -1)], dtype=np.int8)
    for i in range(len(directions)):
        dr = directions[i, 0]; dc = directions[i, 1]; count = 1
        for j in range(1, win_target):
            nr = r + j * dr; nc = c + j * dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                 idx = nr * board_size + nc
                 if idx >= 0 and idx < len(board_flat) and board_flat[idx] == player: count += 1
                 else: break
            else: break
        for j in range(1, win_target):
            nr = r - j * dr; nc = c - j * dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                 idx = nr * board_size + nc
                 if idx >= 0 and idx < len(board_flat) and board_flat[idx] == player: count += 1
                 else: break
            else: break
        if count >= win_target: return True
    return False

check_move_sig = types.UniTuple(boolean, 2)(int8[:], intp, intp, intp, intp, intp)
@njit(check_move_sig, cache=True, fastmath=True, nogil=True)
def _check_move_win_or_block_numba(board_flat, board_size, win_target, r, c, player):
    opponent = -player; is_win = False; is_block = False; idx = r * board_size + c
    if not (0 <= idx < board_size * board_size) or board_flat[idx] != EMPTY: return False, False
    board_flat[idx] = player
    if _check_win_around_pos_numba(board_flat, board_size, win_target, r, c, player): is_win = True
    board_flat[idx] = EMPTY
    if not is_win:
        board_flat[idx] = opponent
        if _check_win_around_pos_numba(board_flat, board_size, win_target, r, c, opponent): is_block = True
        board_flat[idx] = EMPTY
    return is_win, is_block

@njit(cache=True, nogil=True)
def _get_forcing_moves_numba(board_flat, board_size, win_target, player):
    valid_indices = _get_valid_move_indices_numba(board_flat, board_size)
    n_valid = len(valid_indices)
    if n_valid == 0:
        empty_intp_array = np.empty(0, dtype=np.intp)
        return (empty_intp_array, empty_intp_array)
    winning_moves = np.empty(n_valid, dtype=np.intp)
    blocking_moves = np.empty(n_valid, dtype=np.intp)
    win_count = 0; block_count = 0
    board_copy = np.copy(board_flat)
    for i in range(n_valid):
        idx = valid_indices[i]
        if idx < 0 or idx >= len(board_copy): continue
        r = idx // board_size; c = idx % board_size
        is_win, is_block = _check_move_win_or_block_numba(board_copy, board_size, win_target, r, c, player)
        if is_win: winning_moves[win_count] = idx; win_count += 1
        elif is_block: blocking_moves[block_count] = idx; block_count += 1
    return (winning_moves[:win_count], blocking_moves[:block_count])

@njit(intp[:](int8[:], intp), cache=True, fastmath=True, nogil=True)
def _get_valid_move_indices_numba(board_flat, board_size):
    n_total = board_size * board_size
    if n_total == 0: return np.empty(0, dtype=np.intp)
    occupied_indices = np.where(board_flat != EMPTY)[0]
    n_occupied = len(occupied_indices)
    if n_occupied == 0:
        center_idx = (board_size // 2) * board_size + (board_size // 2)
        if center_idx >= 0 and center_idx < n_total: return np.array([center_idx], dtype=np.intp)
        else: return np.empty(0, dtype=np.intp)
    is_neighbor = np.zeros(n_total, dtype=numba.boolean)
    radius = 2
    for occ_idx in occupied_indices:
        if occ_idx < 0 or occ_idx >= n_total: continue
        r_occ = occ_idx // board_size; c_occ = occ_idx % board_size
        r_min = max(0, r_occ - radius); r_max = min(board_size, r_occ + radius + 1)
        c_min = max(0, c_occ - radius); c_max = min(board_size, c_occ + radius + 1)
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                n_idx = r * board_size + c
                if n_idx >= 0 and n_idx < n_total:
                    is_neighbor[n_idx] = True
    n_empty_max = n_total - n_occupied
    potential_indices = np.empty(n_empty_max, dtype=np.intp)
    count = 0
    for idx in range(n_total):
        if is_neighbor[idx] and board_flat[idx] == EMPTY:
            if count < n_empty_max:
                 potential_indices[count] = idx
                 count += 1
    if count == 0:
         empty_indices = np.where(board_flat == EMPTY)[0]
         if len(empty_indices) > 0: return empty_indices
         else: return np.empty(0, dtype=np.intp)
    return potential_indices[:count]

@njit(cache=True, nogil=True)
def _insertion_sort_indices(neg_types, neg_scores, indices_to_sort):
    n = len(indices_to_sort)
    if n <= 1: return indices_to_sort
    for i in range(1, n):
        current_original_index = indices_to_sort[i]
        current_key1 = neg_types[current_original_index]
        current_key2 = neg_scores[current_original_index]
        j = i - 1
        while j >= 0:
            prev_original_index = indices_to_sort[j]
            prev_key1 = neg_types[prev_original_index]
            prev_key2 = neg_scores[prev_original_index]
            if current_key1 < prev_key1 or \
               (current_key1 == prev_key1 and current_key2 < prev_key2):
                indices_to_sort[j + 1] = prev_original_index; j -= 1
            else: break
        indices_to_sort[j + 1] = current_original_index
    return indices_to_sort

@njit(intp[:](int8[:], intp, intp, intp, float64[:], killer_tuple_type, intp, float64, float64, float64), cache=True, nogil=True)
def _get_ordered_moves_numba_opt(
    board_flat, board_size, win_target, player, pos_weights_flat,
    killer_moves_indices, tt_hint_idx, heuristic_pos_weight_scale,
    w_win: float, w_block_win: float
):
    valid_move_indices = _get_valid_move_indices_numba(board_flat, board_size)
    n_valid = len(valid_move_indices)
    if n_valid == 0: return np.empty(0, dtype=np.intp)
    move_types = np.empty(n_valid, dtype=np.int8)
    move_heuristic_scores = np.empty(n_valid, dtype=np.float64)
    win_indices = np.empty(n_valid, dtype=np.intp)
    block_indices = np.empty(n_valid, dtype=np.intp)
    other_indices = np.empty(n_valid, dtype=np.intp)
    win_write_idx = 0; block_write_idx = 0; other_write_idx = 0
    board_copy_for_check = np.copy(board_flat)
    for i in range(n_valid):
        idx = valid_move_indices[i]
        r = idx // board_size; c = idx % board_size
        is_win, is_block = _check_move_win_or_block_numba(board_copy_for_check, board_size, win_target, r, c, player)
        if is_win:
            move_types[i] = 2; move_heuristic_scores[i] = w_win
            win_indices[win_write_idx] = i; win_write_idx += 1
        elif is_block:
            move_types[i] = 1; move_heuristic_scores[i] = w_block_win
            block_indices[block_write_idx] = i; block_write_idx += 1
        else:
            move_types[i] = 0
            move_heuristic_scores[i] = pos_weights_flat[idx] * heuristic_pos_weight_scale
            other_indices[other_write_idx] = i; other_write_idx += 1
    ordered_indices = np.empty(n_valid, dtype=np.intp)
    write_idx = 0
    seen = {np.intp(-1)}
    if win_write_idx > 0:
        is_hint_a_winner = False
        if tt_hint_idx != -1:
            for i in range(win_write_idx):
                 original_idx_in_valid = win_indices[i]
                 actual_move_idx = valid_move_indices[original_idx_in_valid]
                 if actual_move_idx == tt_hint_idx:
                      is_hint_a_winner = True; ordered_indices[write_idx] = tt_hint_idx; seen.add(tt_hint_idx); write_idx += 1
                      break
        for i in range(win_write_idx):
            original_idx_in_valid = win_indices[i]
            actual_move_idx = valid_move_indices[original_idx_in_valid]
            if actual_move_idx not in seen: ordered_indices[write_idx] = actual_move_idx; seen.add(actual_move_idx); write_idx += 1
        if win_write_idx == n_valid: return ordered_indices[:write_idx]
    if tt_hint_idx != -1 and tt_hint_idx not in seen:
        is_valid_tt_hint = False
        for i in range(n_valid):
             if valid_move_indices[i] == tt_hint_idx: is_valid_tt_hint = True; break
        if is_valid_tt_hint: ordered_indices[write_idx] = tt_hint_idx; seen.add(tt_hint_idx); write_idx += 1
    for k_idx in killer_moves_indices:
        if k_idx != -1 and k_idx not in seen:
             is_valid_killer = False
             for i in range(n_valid):
                 if valid_move_indices[i] == k_idx: is_valid_killer = True; break
             if is_valid_killer: ordered_indices[write_idx] = k_idx; seen.add(k_idx); write_idx += 1
    if block_write_idx > 0:
         for i in range(block_write_idx):
             original_idx_in_valid = block_indices[i]
             actual_move_idx = valid_move_indices[original_idx_in_valid]
             if actual_move_idx not in seen: ordered_indices[write_idx] = actual_move_idx; seen.add(actual_move_idx); write_idx += 1
    if other_write_idx > 0:
        remaining_original_indices = np.empty(other_write_idx, dtype=np.intp)
        remaining_count = 0
        for i in range(other_write_idx):
            original_idx_in_valid = other_indices[i]
            actual_move_idx = valid_move_indices[original_idx_in_valid]
            if actual_move_idx not in seen: remaining_original_indices[remaining_count] = original_idx_in_valid; remaining_count += 1
        if remaining_count > 0:
            remaining_original_indices_sliced = remaining_original_indices[:remaining_count]
            remaining_neg_types = np.zeros(n_valid, dtype=np.int8)
            remaining_neg_scores = -move_heuristic_scores
            sorted_remaining_original_indices = _insertion_sort_indices(
                remaining_neg_types, remaining_neg_scores, remaining_original_indices_sliced
            )
            for i in range(remaining_count):
                original_idx = sorted_remaining_original_indices[i]
                ordered_indices[write_idx] = valid_move_indices[original_idx]
                write_idx += 1
    return ordered_indices[:write_idx]

def numba_warmup():
    print("[Numba Warm-up] Triggering JIT compilation...")
    start_warmup_time = time.time()
    dummy_board_size = BOARD_SIZE; dummy_win_target = WIN_TARGET
    dummy_board = np.zeros(dummy_board_size * dummy_board_size, dtype=np.int8)
    center_idx = (dummy_board_size // 2) * dummy_board_size + (dummy_board_size // 2)
    if center_idx >= 0 and center_idx + 1 < len(dummy_board):
         dummy_board[center_idx] = PLAYER_X; dummy_board[center_idx+1] = PLAYER_O
    dummy_r, dummy_c = dummy_board_size // 2, dummy_board_size // 2
    dummy_player = PLAYER_X; dummy_pos_weights = positional_weights_flat
    dummy_weights_tuple = tuple(WEIGHTS_TUPLE_LIST) # Use the latest list
    dummy_killer_moves_indices = tuple([-1] * MAX_KILLER_MOVES)
    dummy_tt_hint = -1; dummy_heuristic_scale = float(EVAL_WEIGHTS['HEURISTIC_POS_WEIGHT_SCALE'])
    dummy_w_win = float(EVAL_WEIGHTS['WIN']); dummy_w_block_win = float(EVAL_WEIGHTS['BLOCK_WIN'])
    try:
        _analyze_line_segment_numba(dummy_board, dummy_board_size, dummy_r, dummy_c, 1, 0, dummy_player, dummy_win_target)
        static_evaluation_numba_core(dummy_board, dummy_board_size, dummy_win_target, dummy_player, dummy_pos_weights, *dummy_weights_tuple)
        _check_win_around_pos_numba(dummy_board, dummy_board_size, dummy_win_target, dummy_r, dummy_c, dummy_player)
        empty_r, empty_c = dummy_r + 1, dummy_c + 1
        if empty_r < dummy_board_size and empty_c < dummy_board_size:
            _check_move_win_or_block_numba(dummy_board, dummy_board_size, dummy_win_target, empty_r, empty_c, dummy_player)
        _get_valid_move_indices_numba(dummy_board, dummy_board_size)
        _get_ordered_moves_numba_opt(dummy_board, dummy_board_size, dummy_win_target, dummy_player, dummy_pos_weights, dummy_killer_moves_indices, dummy_tt_hint, dummy_heuristic_scale, dummy_w_win, dummy_w_block_win)
        _get_forcing_moves_numba(dummy_board, dummy_board_size, dummy_win_target, dummy_player)
        dummy_neg_types = np.array([0, -1, -2], dtype=np.int8); dummy_neg_scores = np.array([-10.0, -5.0, -20.0], dtype=np.float64)
        dummy_indices = np.array([0, 1, 2], dtype=np.intp); _insertion_sort_indices(dummy_neg_types, dummy_neg_scores, dummy_indices)
        end_warmup_time = time.time()
        print(f"[Numba Warm-up] Compilation finished in {end_warmup_time - start_warmup_time:.3f} seconds.")
    except Exception as e:
        print(f"[Numba Warm-up] Error during warm-up: {e}"); traceback.print_exc()

class GomokuEnv:
    def __init__(self, board_size=BOARD_SIZE, win_target=WIN_TARGET, initial_board=None, initial_player=PLAYER_X):
        self.board_size = board_size; self.win_target = win_target
        self.zobrist_table = ZOBRIST_TABLE; self.zobrist_side = ZOBRIST_SIDE_TO_MOVE
        self.board = np.zeros(board_size * board_size, dtype=np.int8)
        self.empty_count = board_size * board_size
        if initial_board is not None:
             if isinstance(initial_board, np.ndarray) and initial_board.ndim == 1 and len(initial_board) == board_size * board_size:
                 self.board = initial_board.astype(np.int8, copy=True)
             else:
                  try:
                      self.board = np.array(initial_board, dtype=np.int8).flatten()
                      if len(self.board) != board_size * board_size: raise ValueError("Initial board has incorrect dimensions.")
                  except Exception as e: raise ValueError(f"Invalid initial_board format: {e}")
             self.current_player = initial_player; self.move_history: List[Tuple[int, int]] = []
             self.current_hash = self._compute_full_hash(); self.winner: Optional[int] = None
             self.empty_count = np.count_nonzero(self.board == EMPTY)
             self.done: bool = self._check_terminal()
        else:
             self.current_player = PLAYER_X; self.move_history = []
             self.current_hash: np.uint64 = np.uint64(0); self.winner = None; self.done = False
             self.reset()

    def _compute_full_hash(self):
        h = np.uint64(0)
        for i in range(self.board_size * self.board_size):
            piece = self.board[i]
            if piece != EMPTY: h ^= self.zobrist_table[i, get_zobrist_piece_index(piece)]
        if self.current_player == PLAYER_O: h ^= self.zobrist_side
        return h

    def _check_terminal(self): # Final version without @profile
        if not self.move_history: self.winner = None; return False
        last_move = self.move_history[-1]; last_player = -self.current_player
        last_r, last_c = last_move
        last_move_idx = last_r * self.board_size + last_c
        if self.check_win(last_player, last_move_idx=last_move_idx):
             self.winner = last_player; return True
        if self.empty_count == 0: # Optimized draw check
            if self.winner is None: self.winner = EMPTY; return True
        self.winner = None; return False

    def reset(self):
        self.board.fill(EMPTY); self.current_player = PLAYER_X; self.done = False
        self.winner = None; self.move_history = []
        self.empty_count = self.board_size * self.board_size
        self.current_hash = self._compute_full_hash()
        return self.board.copy()

    def get_valid_moves(self, check_terminal=False) -> np.ndarray:
         if not check_terminal and self.done: return np.empty(0, dtype=np.intp)
         board_to_check = np.ascontiguousarray(self.board, dtype=np.int8)
         valid_indices = _get_valid_move_indices_numba(board_to_check, self.board_size)
         return valid_indices

    def check_win(self, player: int, last_move_idx: Optional[int] = None) -> bool:
        if player == EMPTY: return False
        board_to_check = np.ascontiguousarray(self.board, dtype=np.int8)
        if last_move_idx is not None:
            if not (0 <= last_move_idx < self.board_size * self.board_size): return False
            if board_to_check[last_move_idx] != player: return False
            r = last_move_idx // self.board_size; c = last_move_idx % self.board_size
            return _check_win_around_pos_numba(board_to_check, self.board_size, self.win_target, r, c, player)
        else:
            if np.sum(board_to_check == player) < self.win_target: return False
            player_indices = np.where(board_to_check == player)[0]
            for idx in player_indices:
                r = idx // self.board_size; c = idx % self.board_size
                if _check_win_around_pos_numba(board_to_check, self.board_size, self.win_target, r, c, player): return True
            return False

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, bool, Optional[int]]:
        if self.done: return self.board.copy(), self.done, self.winner
        row, col = action; idx = row * self.board_size + col
        if not (0 <= idx < self.board_size * self.board_size) or self.board[idx] != EMPTY:
            print(f"错误: 尝试非法移动 {action} 到非空或越界位置。游戏终止，对方获胜。")
            self.done = True; self.winner = -self.current_player; return self.board.copy(), self.done, self.winner
        player_who_moved = self.current_player
        self.empty_count -= 1
        old_piece_z_idx = get_zobrist_piece_index(self.board[idx])
        new_piece_z_idx = get_zobrist_piece_index(player_who_moved)
        self.current_hash ^= self.zobrist_table[idx, old_piece_z_idx] ^ self.zobrist_table[idx, new_piece_z_idx] ^ self.zobrist_side
        self.board[idx] = player_who_moved; self.move_history.append(action)
        # ADDED Immediate win check here for robustness
        if self.check_win(player_who_moved, last_move_idx=idx):
             self.winner = player_who_moved
             self.done = True
             return self.board.copy(), self.done, self.winner
        self.done = self._check_terminal()
        if not self.done: self.current_player *= -1
        return self.board.copy(), self.done, self.winner

    def undo_move(self):
        if not self.move_history: return False
        row, col = self.move_history.pop(); idx = row * self.board_size + col
        if not (0 <= idx < self.board_size * self.board_size): print(f"错误: 尝试撤销无效索引 {idx} 的走法。"); return False
        player_who_moved = self.board[idx]
        if player_who_moved == EMPTY:
            print(f"错误: 尝试撤销位置 {(row,col)} 的走法，但该位置为空。"); self.move_history.append((row, col)); return False
        self.empty_count += 1
        current_piece_z_idx = get_zobrist_piece_index(player_who_moved)
        empty_piece_z_idx = get_zobrist_piece_index(EMPTY)
        self.current_hash ^= self.zobrist_table[idx, current_piece_z_idx] ^ self.zobrist_table[idx, empty_piece_z_idx] ^ self.zobrist_side
        self.board[idx] = EMPTY; self.current_player *= -1
        self.done = False; self.winner = None
        return True

    def get_hash(self) -> np.uint64: return self.current_hash

    def render_text(self):
        header = "   ";
        for i in range(self.board_size): header += f"{i:<2}"
        print(header); print("  +" + "--"*self.board_size + "-+")
        for r in range(self.board_size):
            row_str = f"{r:<2}|"
            for c in range(self.board_size):
                piece = self.board[r * self.board_size + c]
                if piece == PLAYER_X: row_str += " X"
                elif piece == PLAYER_O: row_str += " O"
                else: row_str += " ."
            row_str += " |"; print(row_str)
        print("  +" + "--"*self.board_size + "-+")
        if not self.done: turn = "X (黑棋)" if self.current_player == PLAYER_X else "O (白棋)"; print(f"轮到: {turn}")
        else:
            if self.winner == PLAYER_X: print("结果: X (黑棋) 获胜!")
            elif self.winner == PLAYER_O: print("结果: O (白棋) 获胜!")
            else: print("结果: 平局!")

class AlphaBetaSearcher:
    def __init__(self, board_state: np.ndarray, current_player: int, board_size:int, win_target:int, max_depth: int, time_limit: float, stop_event = None):
        self.env = GomokuEnv(board_size, win_target, initial_board=board_state, initial_player=current_player)
        self.max_depth = max_depth; self.time_limit = time_limit
        self.node_count = 0; self.qnode_count = 0; self.tt_hits = 0
        self.transposition_table: Dict[np.uint64, Tuple[int, float, int, Optional[int], Optional[float]]] = {}
        self.start_time = 0.0; self.time_up = False
        self.stop_event = stop_event
        self.best_move_previous_iter_idx: Optional[int] = None
        self.current_search_depth = 0
        self.killer_moves = np.full((max_depth + QUIESCENCE_SEARCH_DEPTH + 1, MAX_KILLER_MOVES), -1, dtype=np.intp)
        self.heuristic_scale = float(EVAL_WEIGHTS['HEURISTIC_POS_WEIGHT_SCALE'])
        self.board_size = board_size

    def _check_stop(self):
        return self.time_up

    def _clear_tt_if_needed(self):
        if len(self.transposition_table) > TT_SIZE_LIMIT * 1.2: self.transposition_table.clear()

    def _add_killer_move(self, depth: int, move_idx: int):
        if move_idx == -1 or not (0 <= depth < len(self.killer_moves)): return
        if move_idx != self.killer_moves[depth, 0]:
            self.killer_moves[depth, 1] = self.killer_moves[depth, 0]; self.killer_moves[depth, 0] = move_idx

    def _get_ordered_moves(self, depth: int, best_move_hint_idx: Optional[int]) -> np.ndarray:
        board_to_check = np.ascontiguousarray(self.env.board, dtype=np.int8)
        tt_hint_idx: int = best_move_hint_idx if best_move_hint_idx is not None else -1
        if 0 <= depth < len(self.killer_moves): killers = self.killer_moves[depth]
        else: killers = np.array([-1] * MAX_KILLER_MOVES, dtype=np.intp)
        killer_tuple = tuple(killers)
        w_win = float(EVAL_WEIGHTS['WIN']); w_block_win = float(EVAL_WEIGHTS['BLOCK_WIN'])
        ordered_indices = _get_ordered_moves_numba_opt(
            board_to_check, self.board_size, self.env.win_target,
            self.env.current_player, positional_weights_flat,
            killer_tuple, tt_hint_idx, self.heuristic_scale,
            w_win, w_block_win)
        return ordered_indices

    def quiescence_search(self, alpha: float, beta: float, current_ply: int, depth: int) -> float: # No @profile
        if self.qnode_count % QSEARCH_TIME_CHECK_INTERVAL == 0:
             if self.time_up or (self.stop_event and self.stop_event.is_set()):
                  self.time_up = True; return 0.0
        self.qnode_count += 1
        board_to_eval = np.ascontiguousarray(self.env.board, dtype=np.int8)
        stand_pat = static_evaluation_numba_core(board_to_eval, self.board_size, self.env.win_target, self.env.current_player, positional_weights_flat, *WEIGHTS_TUPLE)
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
        if depth <= 0: return alpha
        winning_moves_idx, blocking_moves_idx = _get_forcing_moves_numba(board_to_eval, self.board_size, self.env.win_target, self.env.current_player)
        best_score = stand_pat
        b_size = self.board_size
        for idx in winning_moves_idx:
            move_r = idx // b_size; move_c = idx % b_size; move = (int(move_r), int(move_c))
            player_before_move = self.env.current_player; self.env.step(move); score = 0.0
            if self.env.done:
                winner = self.env.winner
                if winner == player_before_move: score = EVAL_WEIGHTS['WIN'] - (current_ply + 1)
                elif winner == EMPTY: score = 0.0
                else: score = -(EVAL_WEIGHTS['WIN'] - (current_ply + 1))
            else: score = -self.quiescence_search(-beta, -alpha, current_ply + 1, depth - 1)
            self.env.undo_move();
            if self.time_up: return 0.0
            best_score = max(best_score, score); alpha = max(alpha, best_score)
            if alpha >= beta: return beta
        for idx in blocking_moves_idx:
            move_r = idx // b_size; move_c = idx % b_size; move = (int(move_r), int(move_c))
            player_before_move = self.env.current_player; self.env.step(move); score = 0.0
            if self.env.done:
                 winner = self.env.winner
                 if winner == player_before_move: score = EVAL_WEIGHTS['WIN'] - (current_ply + 1)
                 elif winner == EMPTY: score = 0.0
                 else: score = -(EVAL_WEIGHTS['WIN'] - (current_ply + 1))
            else: score = -self.quiescence_search(-beta, -alpha, current_ply + 1, depth - 1)
            self.env.undo_move();
            if self.time_up: return 0.0
            best_score = max(best_score, score); alpha = max(alpha, best_score)
            if alpha >= beta: return beta
        return alpha

    def pvs(self, depth: int, alpha: float, beta: float, maximizing_player: bool, current_ply: int) -> float: # No @profile
        if self.node_count % 1024 == 0:
             if self.time_up or (self.stop_event and self.stop_event.is_set()):
                  self.time_up = True; return 0.0
             if time.time() - self.start_time > self.time_limit: self.time_up = True; return 0.0
        if self.env.done:
            winner = self.env.winner; current_node_player = self.env.current_player; opponent_player = -current_node_player
            if winner == opponent_player: return -(EVAL_WEIGHTS['WIN'] - current_ply)
            elif winner == current_node_player: return EVAL_WEIGHTS['WIN'] - current_ply
            elif winner == EMPTY: return 0.0
            else: return 0.0
        if depth <= 0: return self.quiescence_search(alpha, beta, current_ply, QUIESCENCE_SEARCH_DEPTH)
        self.node_count += 1; original_alpha = alpha
        board_hash = self.env.get_hash(); tt_entry = self.transposition_table.get(board_hash)
        best_move_hint_idx: Optional[int] = None; tt_move_idx: int = -1
        if tt_entry:
            if len(tt_entry) >= 4:
                stored_depth, stored_score, stored_flag, stored_best_move_idx = tt_entry[:4]
                if stored_depth >= depth:
                    self.tt_hits += 1; score = stored_score
                    if stored_flag == TT_ENTRY_FLAGS['EXACT']: return score
                    elif stored_flag == TT_ENTRY_FLAGS['LOWER']: alpha = max(alpha, score)
                    elif stored_flag == TT_ENTRY_FLAGS['UPPER']: beta = min(beta, score)
                    if alpha >= beta: return score
                if stored_best_move_idx is not None and stored_best_move_idx != -1:
                    if 0 <= stored_best_move_idx < self.board_size**2 and self.env.board[stored_best_move_idx] == EMPTY:
                         tt_move_idx = stored_best_move_idx; best_move_hint_idx = tt_move_idx
        ordered_move_indices = self._get_ordered_moves(depth=current_ply, best_move_hint_idx=best_move_hint_idx)
        if len(ordered_move_indices) == 0: return 0.0
        best_score = -INF; move_idx_for_tt: int = -1
        player_at_this_node = self.env.current_player
        b_size = self.board_size
        for i, current_move_idx in enumerate(ordered_move_indices):
            move_r = current_move_idx // b_size; move_c = current_move_idx % b_size; move = (int(move_r), int(move_c))
            self.env.step(move); eval_score = 0.0
            if self.env.done:
                 winner = self.env.winner
                 if winner == player_at_this_node: eval_score = EVAL_WEIGHTS['WIN'] - (current_ply + 1)
                 elif winner == EMPTY: eval_score = 0.0
                 else: eval_score = -(EVAL_WEIGHTS['WIN'] - (current_ply + 1))
            else:
                if i == 0: eval_score = -self.pvs(depth - 1, -beta, -alpha, not maximizing_player, current_ply + 1)
                else:
                    eval_score = -self.pvs(depth - 1, -alpha - 1e-9, -alpha, not maximizing_player, current_ply + 1)
                    if alpha < eval_score < beta: eval_score = -self.pvs(depth - 1, -beta, -alpha, not maximizing_player, current_ply + 1)
            self.env.undo_move()
            if self.time_up: return 0.0
            if eval_score > best_score: best_score = eval_score; move_idx_for_tt = current_move_idx
            alpha = max(alpha, best_score)
            if alpha >= beta:
                if current_move_idx != tt_move_idx: self._add_killer_move(current_ply, current_move_idx)
                break
        final_score = best_score; tt_flag = TT_ENTRY_FLAGS['EXACT']
        if final_score <= original_alpha: tt_flag = TT_ENTRY_FLAGS['UPPER']
        elif final_score >= beta: tt_flag = TT_ENTRY_FLAGS['LOWER']
        existing_depth = -1
        if tt_entry and len(tt_entry) > 0 : existing_depth = tt_entry[0]
        if depth >= existing_depth:
            if len(self.transposition_table) < TT_SIZE_LIMIT:
                self.transposition_table[board_hash] = (depth, final_score, tt_flag, move_idx_for_tt, None)
        return final_score

    def find_best_move_iddfs(self) -> Optional[Tuple[int, int]]:
        self.node_count = 0; self.qnode_count = 0; self.tt_hits = 0
        self.start_time = time.time(); self.time_up = False
        self._clear_tt_if_needed(); self.killer_moves.fill(-1)
        root_player = self.env.current_player; is_maximizing_root = (root_player == PLAYER_X)
        best_move_overall_idx: Optional[int] = None; best_score_overall = -INF
        self.best_move_previous_iter_idx = None
        if self.env.done: return None
        b_size = self.board_size
        for depth in range(1, self.max_depth + 1):
            self.current_search_depth = depth; search_start_depth_time = time.time()
            alpha = -INF; beta = INF
            ordered_move_indices = self._get_ordered_moves(depth=0, best_move_hint_idx=self.best_move_previous_iter_idx)
            if len(ordered_move_indices) == 0 : break
            current_best_score_for_depth = -INF
            if not ordered_move_indices.size: continue
            current_best_move_idx_for_depth = ordered_move_indices[0]
            for i, current_move_idx in enumerate(ordered_move_indices):
                if self.time_up: break
                move_r = current_move_idx // b_size; move_c = current_move_idx % b_size; move = (int(move_r), int(move_c))
                self.env.step(move); eval_score = 0.0
                if self.env.done:
                    winner = self.env.winner
                    if winner == root_player: eval_score = EVAL_WEIGHTS['WIN'] - 1
                    elif winner == EMPTY: eval_score = 0.0
                    else: eval_score = -EVAL_WEIGHTS['WIN'] + 1
                else:
                    if i == 0: eval_score = -self.pvs(depth - 1, -beta, -alpha, not is_maximizing_root, 1)
                    else:
                        eval_score = -self.pvs(depth - 1, -alpha - 1e-9, -alpha, not is_maximizing_root, 1)
                        if alpha < eval_score < beta: eval_score = -self.pvs(depth - 1, -beta, -alpha, not is_maximizing_root, 1)
                self.env.undo_move()
                if self.time_up: break
                if eval_score > current_best_score_for_depth:
                    current_best_score_for_depth = eval_score; current_best_move_idx_for_depth = current_move_idx
                alpha = max(alpha, current_best_score_for_depth)
            search_end_depth_time = time.time(); depth_time = search_end_depth_time - search_start_depth_time; total_time = search_end_depth_time - self.start_time
            if not self.time_up:
                best_move_overall_idx = current_best_move_idx_for_depth
                best_score_overall = current_best_score_for_depth
                self.best_move_previous_iter_idx = best_move_overall_idx
                best_move_print_r = best_move_overall_idx // b_size; best_move_print_c = best_move_overall_idx % b_size
                best_move_print = (int(best_move_print_r), int(best_move_print_c))
                score_str = f"{best_score_overall:.1f}";
                if abs(best_score_overall) > 1e6: score_str = f"{best_score_overall:.2e}"
                print(f"深度 {depth}: 最佳走法 {best_move_print} (Idx:{best_move_overall_idx}) 分数 {score_str} "
                      f"(耗时 {depth_time:.3f}s, 总计 {total_time:.3f}s, "
                      f"节点 {self.node_count}, Q节点 {self.qnode_count}, TT命中 {self.tt_hits})")
                if abs(best_score_overall) >= EVAL_WEIGHTS['WIN'] * 0.95: print(f"深度 {depth}: 找到必胜/必败走法，提前结束搜索。"); break
            else: print(f"深度 {depth}: 超时! 使用上一深度结果。 (总计 {total_time:.3f}s)"); break
            if time.time() - self.start_time > self.time_limit:
                 if not self.time_up: print(f"深度 {depth} 完成后超时。使用当前深度结果。"); break
        final_move_tuple: Optional[Tuple[int, int]] = None
        if best_move_overall_idx is None:
            print("警告: 未找到最佳走法 (可能超时或无有效走法)。选择随机有效走法。")
            valid_move_indices = self.env.get_valid_moves()
            if len(valid_move_indices) > 0:
                best_move_overall_idx = random.choice(valid_move_indices)
                final_r = best_move_overall_idx // b_size; final_c = best_move_overall_idx % b_size
                final_move_tuple = (int(final_r), int(final_c))
            else: return None
        else:
            final_r = best_move_overall_idx // b_size; final_c = best_move_overall_idx % b_size
            final_move_tuple = (int(final_r), int(final_c))
        final_depth = self.current_search_depth if not self.time_up else self.current_search_depth - 1
        final_depth = max(1, final_depth); final_score_str = f"{best_score_overall:.1f}";
        if abs(best_score_overall) > 1e6: final_score_str = f"{best_score_overall:.2e}"
        print(f"AI 最终选择: {final_move_tuple} (来自深度 {final_depth}, 分数: {final_score_str})")
        return final_move_tuple

class AlphaBetaPlayer:
    def __init__(self, board_state: np.ndarray, current_player: int, board_size:int, win_target:int, search_depth: int, time_limit: float):
        self.searcher = AlphaBetaSearcher(board_state, current_player, board_size, win_target, search_depth, time_limit, stop_event=None)

    def get_action(self) -> Optional[Tuple[int, int]]:
        best_move = self.searcher.find_best_move_iddfs()
        return best_move

# --- Game Modes ---
def run_single_game(game_index: int, search_depth: int, time_limit: float, board_size: int, win_target: int) -> Optional[Tuple[List[Tuple[int, int]], Optional[int]]]:
    env = GomokuEnv(board_size=board_size, win_target=win_target); turn_count = 0; max_turns = board_size * board_size + 5
    try:
        while not env.done and turn_count < max_turns:
            turn_count += 1; current_player_id = env.current_player
            current_player_agent = AlphaBetaPlayer(env.board.copy(), current_player_id, board_size, win_target, search_depth, time_limit)
            action = current_player_agent.get_action()
            if action is None: print(f"游戏 {game_index+1}, 回合 {turn_count}: 玩家 {current_player_id} 未能找到走法!"); env.winner = -current_player_id; env.done = True; break
            _, _, _ = env.step(action)
        if not env.done:
            if turn_count >= max_turns: print(f"游戏 {game_index+1}: 达到最大回合数 {max_turns}，判定为平局。"); env.winner = EMPTY; env.done = True
            else:
                 env.done = env._check_terminal();
                 if not env.done: env.winner = EMPTY; env.done = True
        return (list(env.move_history), env.winner)
    except Exception as e: print(f"!!! 运行 run_single_game (局 {game_index+1}) 时发生错误: {e}"); traceback.print_exc(); return None

def self_play_mode(num_games: int, output_file: str, search_depth: int = DEFAULT_SEARCH_DEPTH, time_limit: float = DEFAULT_TIME_LIMIT_SECONDS):
    print(f"\n--- 开始自我对弈模式 (并行 Pool, {num_games} 局) ---")
    print(f"棋盘: {BOARD_SIZE}x{BOARD_SIZE}, 获胜: {WIN_TARGET}, 深度: {search_depth}, 时限: {time_limit}s, 输出文件: {output_file}")
    num_workers = min(num_games, os.cpu_count() or 1); print(f"使用 {num_workers} 个进程并行处理...")
    game_data: List[Tuple[List[Tuple[int, int]], Optional[int]]] = []; wins = {PLAYER_X: 0, PLAYER_O: 0, EMPTY: 0}
    start_time_total = time.time(); game_args = [(i, search_depth, time_limit, BOARD_SIZE, WIN_TARGET) for i in range(num_games)]; results = []
    try:
        with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool: results = pool.starmap(run_single_game, game_args)
    except Exception as e: print(f"!!! 多进程池运行时发生错误: {e}"); traceback.print_exc()
    successful_games = 0
    for i, result in enumerate(results):
        if result is not None:
            history, winner = result
            if isinstance(history, list) and winner in wins: game_data.append((history, winner)); wins[winner] += 1; successful_games += 1
            else: print(f"警告: 游戏 {i+1} 返回了无效结果: history_type={type(history)}, winner={winner}")
        else: print(f"游戏 {i+1} 执行失败，已跳过。")
    end_time_total = time.time(); total_duration = end_time_total - start_time_total; avg_time_per_game = total_duration / successful_games if successful_games > 0 else 0
    print(f"\n--- 自我对弈 (并行 Pool) 结束 ---")
    print(f"成功生成 {successful_games}/{num_games} 局游戏数据，总耗时 {total_duration:.2f} 秒 ({avg_time_per_game:.2f} 秒/局)。")
    print(f"统计: X 胜 {wins[PLAYER_X]} 局, O 胜 {wins[PLAYER_O]} 局, 平局 {wins[EMPTY]} 局。")
    if game_data:
        try:
            with open(output_file, 'wb') as f: pickle.dump(game_data, f); print(f"数据已保存到 {output_file}")
        except Exception as e: print(f"错误：保存数据到 {output_file} 失败: {e}")
    else: print("没有成功生成游戏数据，未保存文件。")

def play_vs_ai_gui(search_depth: int = DEFAULT_SEARCH_DEPTH, time_limit: float = DEFAULT_TIME_LIMIT_SECONDS):
    global pygame
    if not PYGAME_AVAILABLE: print("错误: Pygame 未安装或无法加载。无法启动 GUI 模式。"); print("请尝试安装: pip install pygame"); return
    try:
        if not pygame.get_init(): pygame.init()
    except Exception as e: print(f"错误: 初始化 Pygame 时出错: {e}"); return
    print(f"\n--- 开始人机对战 (GUI 模式, {BOARD_SIZE}x{BOARD_SIZE}) ---")
    env = GomokuEnv(board_size=BOARD_SIZE, win_target=WIN_TARGET)
    human_player = PLAYER_X; ai_player_id = PLAYER_O
    try:
        color_choice = input("选择你的颜色 (X=黑棋, O=白棋, 默认 X): ").upper().strip()
        if color_choice == 'O': human_player = PLAYER_O; ai_player_id = PLAYER_X; print(f"好的, 你执 白棋 (O)。 AI 执 黑棋 (X)。 AI先行。")
        else: human_player = PLAYER_X; ai_player_id = PLAYER_O; print(f"好的, 你执 黑棋 (X)。 AI 执 白棋 (O)。 你先行。")
    except (EOFError, KeyboardInterrupt): print("\n颜色选择被中断。退出游戏。"); pygame.quit(); return
    pygame.display.set_caption(f"五子棋 AI ({BOARD_SIZE}x{BOARD_SIZE})")
    SQUARE_SIZE = 40; MARGIN = 20; INFO_HEIGHT = 50; WIDTH = BOARD_SIZE * SQUARE_SIZE + MARGIN * 2; HEIGHT = BOARD_SIZE * SQUARE_SIZE + MARGIN * 2 + INFO_HEIGHT
    size = (WIDTH, HEIGHT); screen = pygame.display.set_mode(size)
    BLACK = (0, 0, 0); WHITE = (255, 255, 255); BOARD_COLOR = (210, 180, 140); LINE_COLOR = (50, 50, 50); PLAYER_X_COLOR = BLACK; PLAYER_O_COLOR = WHITE
    HIGHLIGHT_COLOR = (255, 255, 0, 120); VALID_MOVE_HOVER_COLOR = (0, 255, 0, 80)
    try: font = pygame.font.SysFont('SimHei', 24); font_large = pygame.font.SysFont('SimHei', 30)
    except: print("警告: 未找到指定中文字体 ('SimHei'), 使用默认字体。"); font = pygame.font.SysFont(None, 30); font_large = pygame.font.SysFont(None, 40)
    def draw_board(surface):
        surface.fill(BOARD_COLOR)
        for i in range(BOARD_SIZE):
            x = MARGIN + i * SQUARE_SIZE + SQUARE_SIZE // 2; start_y = MARGIN + SQUARE_SIZE // 2; end_y = HEIGHT - MARGIN - SQUARE_SIZE // 2 - INFO_HEIGHT; pygame.draw.line(surface, LINE_COLOR, (x, start_y), (x, end_y), 1)
            y = MARGIN + i * SQUARE_SIZE + SQUARE_SIZE // 2; start_x = MARGIN + SQUARE_SIZE // 2; end_x = WIDTH - MARGIN - SQUARE_SIZE // 2; pygame.draw.line(surface, LINE_COLOR, (start_x, y), (end_x, y), 1)
        radius = SQUARE_SIZE // 2 - 3
        if env.move_history:
            lr, lc = env.move_history[-1]; last_move_rect = pygame.Rect(MARGIN + lc * SQUARE_SIZE, MARGIN + lr * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            try: highlight_surf = pygame.Surface(last_move_rect.size, pygame.SRCALPHA); highlight_surf.fill(HIGHLIGHT_COLOR); surface.blit(highlight_surf, last_move_rect.topleft)
            except pygame.error as e: print(f"绘制高亮时出错 (可能 surface 不支持 alpha): {e}")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                idx = r * BOARD_SIZE + c; piece = env.board[idx]; center_x = MARGIN + c * SQUARE_SIZE + SQUARE_SIZE // 2; center_y = MARGIN + r * SQUARE_SIZE + SQUARE_SIZE // 2
                if piece == PLAYER_X: pygame.draw.circle(surface, PLAYER_X_COLOR, (center_x, center_y), radius)
                elif piece == PLAYER_O: pygame.draw.circle(surface, PLAYER_O_COLOR, (center_x, center_y), radius); pygame.draw.circle(surface, BLACK, (center_x, center_y), radius, 1)
    def draw_info(surface, message): pygame.draw.rect(surface, BLACK, (0, HEIGHT - INFO_HEIGHT, WIDTH, INFO_HEIGHT)); text_surf = font.render(message, True, WHITE); text_rect = text_surf.get_rect(center=(WIDTH // 2, HEIGHT - INFO_HEIGHT // 2)); surface.blit(text_surf, text_rect)
    def get_row_col_from_mouse(pos):
        x, y = pos; board_area_width = BOARD_SIZE * SQUARE_SIZE; board_area_height = BOARD_SIZE * SQUARE_SIZE
        if MARGIN + SQUARE_SIZE//4 < x < MARGIN + board_area_width - SQUARE_SIZE//4 and MARGIN + SQUARE_SIZE//4 < y < MARGIN + board_area_height - SQUARE_SIZE//4 :
            c = int((x - MARGIN) // SQUARE_SIZE); r = int((y - MARGIN) // SQUARE_SIZE)
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE: return r, c
        return None
    running = True; game_over = False; status_message = ""; ai_thinking = False; human_turn_active = False
    env.reset(); start_player_id = env.current_player; start_player_color = "黑棋 (X)" if start_player_id == PLAYER_X else "白棋 (O)"
    if start_player_id == human_player: print(f"游戏开始，由 你 ({start_player_color}) 先手。"); status_message = f"你的回合 ({start_player_color})"; human_turn_active = True
    else: print(f"游戏开始，由 AI ({start_player_color}) 先手。"); status_message = f"AI 的回合 ({start_player_color})"; ai_thinking = True; human_turn_active = False
    clock = pygame.time.Clock(); ai_player_agent = None
    while running:
        human_turn_active = (env.current_player == human_player) and not env.done and not ai_thinking
        mouse_pos = pygame.mouse.get_pos(); hover_coords = get_row_col_from_mouse(mouse_pos) if human_turn_active else None
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN and human_turn_active:
                coords = get_row_col_from_mouse(event.pos)
                if coords:
                    r, c = coords; idx = r * env.board_size + c
                    if 0 <= idx < env.board_size * env.board_size and env.board[idx] == EMPTY:
                        action = coords; _, game_done, winner = env.step(action); human_turn_active = False
                        if game_done: game_over = True
                        else: status_message = f"AI 的回合 ({'X' if ai_player_id == PLAYER_X else 'O'})"; ai_thinking = True
                    else: status_message = "无效移动 (位置已被占据或无效)，请重试。"
                else: status_message = "点击位置不在棋盘网格内。"
        if not env.done and env.current_player == ai_player_id and not ai_thinking: ai_thinking = True
        if ai_thinking and not game_over:
            draw_board(screen); msg = f"AI ({'X' if ai_player_id == PLAYER_X else 'O'}) 正在思考..."; draw_info(screen, msg); pygame.display.flip()
            ai_player_agent = AlphaBetaPlayer(env.board.copy(), ai_player_id, env.board_size, env.win_target, search_depth, time_limit)
            action = ai_player_agent.get_action(); ai_thinking = False
            if action is not None:
                _, game_done, winner = env.step(action)
                if game_done: game_over = True
                else: status_message = f"你的回合 ({'X' if human_player == PLAYER_X else 'O'})"; human_turn_active = True
            else: print("AI 未能找到合适的移动!"); status_message = "AI 错误，游戏可能结束。"; env.winner = human_player; env.done = True; game_over = True
        draw_board(screen)
        if human_turn_active and hover_coords:
            hr, hc = hover_coords; h_idx = hr * env.board_size + hc;
            if 0 <= h_idx < env.board_size * env.board_size and env.board[h_idx] == EMPTY:
                hover_rect = pygame.Rect(MARGIN + hc * SQUARE_SIZE, MARGIN + hr * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                try: hover_surf = pygame.Surface(hover_rect.size, pygame.SRCALPHA); hover_surf.fill(VALID_MOVE_HOVER_COLOR); screen.blit(hover_surf, hover_rect.topleft)
                except pygame.error as e: print(f"绘制悬停时出错: {e}")
        current_status = status_message
        if env.done:
             game_over = True
             winner = env.winner
             if winner == PLAYER_X: current_status = "游戏结束: 黑棋 (X) 获胜!"
             elif winner == PLAYER_O: current_status = "游戏结束: 白棋 (O) 获胜!"
             else: current_status = "游戏结束: 平局!"
        elif ai_thinking: current_status = f"AI ({'X' if ai_player_id == PLAYER_X else 'O'}) 正在思考..."
        elif not env.done: player_name = "你" if env.current_player == human_player else "AI"; player_color = "X" if env.current_player == PLAYER_X else "O"; current_status = f"{player_name}的回合 ({player_color})"
        draw_info(screen, current_status); pygame.display.flip()
        if game_over: print(current_status); pygame.time.wait(3000); running = False
        clock.tick(30)
    pygame.quit()
    print(f"\n--- 游戏结束 (GUI 模式, {BOARD_SIZE}x{BOARD_SIZE}) ---")

def profile_single_ai_move(search_depth: int = DEFAULT_SEARCH_DEPTH, time_limit: float = DEFAULT_TIME_LIMIT_SECONDS):
    numba_warmup()
    print(f"\n--- 开始分析单次 AI 决策 ---")
    print(f"棋盘: {BOARD_SIZE}x{BOARD_SIZE}, 获胜: {WIN_TARGET}, 深度: {search_depth}, 时限: {time_limit}s")
    env = GomokuEnv(board_size=BOARD_SIZE, win_target=WIN_TARGET)
    center_r, center_c = BOARD_SIZE // 2, BOARD_SIZE // 2
    try:
        if env.board[center_r * BOARD_SIZE + center_c] == EMPTY: env.step((center_r, center_c)); print(f"模拟下子: ({center_r},{center_c}) by X")
        if not env.done and env.board[(center_r+1) * BOARD_SIZE + center_c] == EMPTY: env.step((center_r+1, center_c)); print(f"模拟下子: ({center_r+1},{center_c}) by O")
        if not env.done and env.board[center_r * BOARD_SIZE + (center_c+1)] == EMPTY: env.step((center_r, center_c+1)); print(f"模拟下子: ({center_r},{center_c+1}) by X")
    except Exception as e: print(f"模拟开局时出错: {e}"); return
    if env.done: print("错误: 模拟开局后游戏意外结束，无法分析。"); return
    print("\n当前棋盘状态 (用于分析):"); env.render_text()
    current_player_id = env.current_player; player_color = "X" if current_player_id == PLAYER_X else "O"
    print(f"\n轮到 {player_color} 思考...")
    ai_player = AlphaBetaPlayer(env.board.copy(), current_player_id, env.board_size, env.win_target, search_depth, time_limit)
    profiler = cProfile.Profile(); print("[性能分析已启用 (单次决策)]"); profiler.enable()
    start_decision_time = time.time(); best_move = ai_player.get_action(); end_decision_time = time.time()
    profiler.disable(); print("[性能分析已禁用]")
    print(f"单次决策耗时: {end_decision_time - start_decision_time:.3f} 秒"); print(f"找到的最佳走法: {best_move}")
    s = io.StringIO(); sortby = 'cumulative'
    try:
        stats = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        print("\n--- cProfile 性能分析结果 (单次 AI 决策, 按累积耗时排序) ---"); stats.print_stats(40); print(s.getvalue()); print("-" * 60)
        print("解读指南:"); print("  ncalls : 函数调用次数。"); print("  tottime: 函数自身运行总时间 (不含调用子函数的时间)。"); print("  cumtime: 函数累计运行总时间 (包含调用子函数的时间)。此列最能反映瓶颈。")
        print("  关注 cumtime 最高的函数，特别是 find_best_move_iddfs, pvs, quiescence_search,"); print("  static_evaluation_numba_core, _get_ordered_moves*, _get_valid_moves* 等。")
        print("  (Numba 编译开销应已通过 warm-up 降低)"); print("-" * 60)
        profile_filename = f"gomoku_single_move_profile_d{search_depth}_t{int(time_limit)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        stats.dump_stats(profile_filename)
        print(f"详细性能分析数据已保存到: {profile_filename}"); print(f"提示: 可以使用 'snakeviz' 可视化 (pip install snakeviz):"); print(f"  snakeviz \"{profile_filename}\"")
        print(f"提示: 也可以使用 'line_profiler' 进行代码行级分析 (需手动添加@profile)。")
    except Exception as dump_e: print(f"错误：保存或打印性能分析数据失败: {dump_e}")
    print("\n--- 单次 AI 决策分析结束 ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    if not NUMBA_AVAILABLE: print("错误：Numba 未找到或导入失败，程序无法继续。"); sys.exit(1)
    numba_warmup()
    print("\n" + "="*50); print(f" 五子棋 AI (IDDFS + PVS + QSearch + Numba)"); print(f" 棋盘: {BOARD_SIZE}x{BOARD_SIZE}, 获胜: {WIN_TARGET} 子连珠"); print("="*50)
    print(f"Numba 加速: {'已启用 (必需)'}"); print(f"Pygame GUI: {'可用' if PYGAME_AVAILABLE else '不可用 (请运行 pip install pygame)'}"); print(f"多进程自我对弈: {'已启用'}")
    action = None
    try:
        while True:
            print("\n选择操作模式:")
            if PYGAME_AVAILABLE: print(f"  1 - 人机对战 (图形界面 - Pygame)")
            else: print(f"  1 - 人机对战 (图形界面 - Pygame) [不可用, 请安装 Pygame]")
            print(f"  2 - AI 自我对弈 (并行 Pool, 生成数据, 带统计)")
            print(f"  3 - 分析单次 AI 决策性能 (cProfile)") # Removed line_profiler mention
            print("  exit - 退出程序")
            choice = input("输入命令或数字: ").lower().strip()
            if choice == 'exit': action = 'exit'; break
            elif choice == '1':
                if PYGAME_AVAILABLE: action = 'play_gui'; break
                else: print("错误: Pygame 不可用。请先安装 Pygame (pip install pygame)。"); continue
            elif choice == '2': action = 'self_play_pool'; break
            elif choice == '3': action = 'profile_single'; break
            else: print("无效命令。")
    except (EOFError, KeyboardInterrupt): action = 'exit'
    if action != 'exit':
        depth = DEFAULT_SEARCH_DEPTH; time_limit = DEFAULT_TIME_LIMIT_SECONDS
        if action in ['play_gui', 'self_play_pool', 'profile_single']:
             print(f"\n使用默认搜索深度 {DEFAULT_SEARCH_DEPTH} 和时限 {DEFAULT_TIME_LIMIT_SECONDS}s。")
             try:
                 depth_str = input(f"输入 AI 最大搜索深度 (整数, 回车使用默认 {DEFAULT_SEARCH_DEPTH}): ").strip()
                 if depth_str: depth = max(1, int(depth_str))
             except ValueError: print(f"输入无效。使用默认深度 {DEFAULT_SEARCH_DEPTH}.")
             except (EOFError, KeyboardInterrupt): print("\n输入已取消。"); action = 'exit'
             if action != 'exit':
                 try:
                     time_str = input(f"输入 AI 每步时间限制 (秒, >= 0.5, 回车使用默认 {DEFAULT_TIME_LIMIT_SECONDS}): ").strip()
                     if time_str: time_limit = max(0.5, float(time_str))
                 except ValueError: print(f"输入无效。使用默认时限 {DEFAULT_TIME_LIMIT_SECONDS}s.")
                 except (EOFError, KeyboardInterrupt): print("\n输入已取消。"); action = 'exit'
    if action == 'play_gui':
         if PYGAME_AVAILABLE:
             try: play_vs_ai_gui(search_depth=depth, time_limit=time_limit)
             except Exception as e:
                  print(f"\n!!! 运行 play_vs_ai_gui 时发生严重错误: {e}"); traceback.print_exc()
                  if 'pygame' in sys.modules and sys.modules['pygame'].get_init():
                      try: pygame.quit()
                      except: pass
    elif action == 'self_play_pool':
        num_games = 10; now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"gomoku_selfplay_{BOARD_SIZE}x{BOARD_SIZE}_d{depth}_t{int(time_limit)}_{now_str}.pkl"
        profile_filename = default_filename.replace('.pkl', '.prof')
        try:
            games_str = input(f"输入自我对弈游戏局数 (整数, 默认 {num_games}): ").strip()
            if games_str: num_games = max(1, int(games_str))
        except ValueError: print(f"输入无效。使用默认 {num_games} 局。")
        except (EOFError, KeyboardInterrupt): print("\n输入已取消。"); action = 'exit'
        if action == 'self_play_pool':
            output_file = input(f"输入保存数据的文件名 (回车使用默认: {default_filename}): ").strip()
            if not output_file: output_file = default_filename
            profile_filename = output_file.replace('.pkl', '.prof')
            enable_profiling_str = input("是否启用性能分析 (y/n, 默认 n)? [分析主进程开销]: ").lower().strip()
            enable_profiling = (enable_profiling_str == 'y')
            profiler = None
            if enable_profiling: print("\n[性能分析已启用 (主进程), 结果保存到 .prof]"); profiler = cProfile.Profile(); profiler.enable()
            else: print("\n[性能分析未启用]")
            try: self_play_mode(num_games, output_file, search_depth=depth, time_limit=time_limit)
            except Exception as e: print(f"\n!!! 运行 self_play_mode 时发生严重错误: {e}"); traceback.print_exc()
            finally:
                if profiler:
                    profiler.disable(); print("[性能分析已禁用]\n"); s = io.StringIO(); sortby = 'cumulative'
                    try:
                        stats = pstats.Stats(profiler, stream=s).sort_stats(sortby)
                        print("--- cProfile 性能分析结果 (主进程 Pool 协调及其他开销, 按累积耗时排序) ---"); stats.print_stats(30); print(s.getvalue()); print("-" * 60)
                        print("解读指南:"); print("  注意: 此分析主要反映主进程活动 (任务创建/管理/通信/结果收集) 的开销，"); print("        **不包含** 子进程中实际 AI 搜索 (`run_single_game` 内部) 的耗时。"); print("-" * 60);
                        stats.dump_stats(profile_filename); print(f"详细性能分析数据已保存到: {profile_filename}"); print(f"提示: 可以使用 'snakeviz' 可视化 (pip install snakeviz):"); print(f"  snakeviz \"{profile_filename}\"")
                    except Exception as dump_e: print(f"错误：保存或打印主进程性能分析数据失败: {dump_e}")
    elif action == 'profile_single':
         try: profile_single_ai_move(search_depth=depth, time_limit=time_limit)
         except Exception as e: print(f"\n!!! 运行 profile_single_ai_move 时发生严重错误: {e}"); traceback.print_exc()
    print("\n--- 程序退出 ---")