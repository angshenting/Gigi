import sys
import ctypes
from typing import Dict, List
from collections import Counter
from objects import Card
from ddsolver import dds
from colorama import Fore, Back, Style, init

init()

# The number of threads is automatically configured by DDS on Windows, taking into account the number of processor cores and available memory.  
# The number of threads can be influenced using by calling `SetMaxThreads`. 
# This function should probably always be called on Linux/Mac, with a zero argument for auto-configuration.
dds.SetMaxThreads(0)

class DDSolver:

    # Default for dds_mode changed to 1
    # Transport table will be reused if same trump suit and the same or nearly the same cards distribution, deal.first can be different. 
    # Always search to find the score. Even when the hand to play has only one card, with possible equivalents, to play.  
    # If zero, we not always find the score
    # If 2 transport tables ignore trump
 
    def __init__(self, dds_mode=1, verbose=False):
        if verbose:
            sys.stderr.write(f"DDSolver being loaded version 2.9.0 - dds mode {dds_mode}\n")
        self.dds_mode = dds_mode
        self.bo = dds.boardsPBN()
        self.solved = dds.solvedBoards()

    def version(self):  
        return "2.9.0"
    
    def calculate_par_batch(self, hands, vulns):
        """Compute PAR scores for multiple deals using CalcAllTablesPBN.

        Args:
            hands: list of PBN hand strings (without 'N:' prefix)
            vulns: list of [vuln_ns, vuln_ew] pairs

        Returns:
            list of int PAR scores (NS perspective), 0 on failure
        """
        if not hands:
            return []

        results = []
        # DDS limit: noOfTables * DDS_STRAINS <= MAXNOOFBOARDS
        max_batch = dds.MAXNOOFBOARDS // dds.DDS_STRAINS  # 200/5 = 40

        for start in range(0, len(hands), max_batch):
            chunk_hands = hands[start:start + max_batch]
            chunk_vulns = vulns[start:start + max_batch]
            chunk_results = self._calc_batch_chunk(chunk_hands, chunk_vulns)
            results.extend(chunk_results)

        return results

    def _calc_batch_chunk(self, hands, vulns):
        """Process a single batch chunk (up to 50 deals) through CalcAllTablesPBN."""
        n = len(hands)
        deals_pbn = dds.ddTableDealsPBN()
        deals_pbn.noOfTables = n

        for i in range(n):
            deals_pbn.deals[i].cards = ("N:" + hands[i]).encode('utf-8')

        table_res = dds.ddTablesRes()
        par_res = dds.allParResults()

        # trumpFilter: all zeros = compute all 5 strains
        trump_filter = (ctypes.c_int * dds.DDS_STRAINS)(0, 0, 0, 0, 0)

        res = dds.CalcAllTablesPBN(
            ctypes.pointer(deals_pbn),
            -1,  # mode: -1 = no par calculation by CalcAllTablesPBN
            trump_filter,
            ctypes.pointer(table_res),
            ctypes.pointer(par_res),
        )

        if res != 1:
            # Fallback: compute individually
            error_message = dds.get_error_message(res)
            sys.stderr.write(f"CalcAllTablesPBN error {res}: {error_message}, falling back to individual\n")
            return [self._single_par(hands[i], vulns[i]) for i in range(n)]

        # Call Par individually on each DD table result
        scores = []
        for i in range(n):
            try:
                table_ptr = ctypes.pointer(table_res.results[i])
                pres = dds.parResults()

                v = 0
                if vulns[i][0]: v = 2
                if vulns[i][1]: v = 3
                if vulns[i][0] and vulns[i][1]: v = 1

                par_ret = dds.Par(table_ptr, pres, v)
                if par_ret != 1:
                    scores.append(0)
                    continue

                par_ptr = ctypes.pointer(pres)
                par_str = par_ptr.contents.parScore[0].value.decode('utf-8')
                ns_score = int(par_str.split()[1])
                scores.append(ns_score)
            except Exception:
                scores.append(0)

        return scores

    def _single_par(self, hand, vuln):
        """Fallback: compute PAR for a single deal."""
        try:
            par = self.calculatepar(hand, vuln, print_result=False)
            return par if par is not None else 0
        except Exception:
            return 0

    def calculatepar(self, hand, vuln, print_result=True):
        tableDealPBN = dds.ddTableDealPBN()
        table = dds.ddTableResults()
        myTable = ctypes.pointer(table)

        line = ctypes.create_string_buffer(80)

        # Need dealer
        tableDealPBN.cards = ("N:"+hand).encode('utf-8')

        res = dds.CalcDDtablePBN(tableDealPBN, myTable)

        if res != 1:
            error_message = dds.get_error_message(res)
            sys.stderr.write(f"Error Code: {res}, Error Message: {error_message}, Hand {hand.encode('utf-8')}\n")
            raise Exception(error_message)

        pres = dds.parResults()

        # vulnerable 
        # 0: None 1: Both 2: NS 3: EW 
        v = 0
        if vuln[0]: v = 2
        if vuln[1]: v = 3
        if vuln[0] and vuln[1]: v = 1

        res = dds.Par(myTable, pres, v)

        if res != 1:
            error_message = dds.get_error_message(res)
            sys.stderr.write(f"{Fore.RED}Error Code: {res}, Error Message: {error_message} {hand.encode('utf-8')}{Style.RESET_ALL}")
            return None

        par = ctypes.pointer(pres)

        if print_result:
            print("NS score: {}".format(par.contents.parScore[0].value.decode('utf-8')))
            print("EW score: {}".format(par.contents.parScore[1].value.decode('utf-8')))
            #print("NS list : {}".format(par.contents.parContractsString[0].value.decode('utf-8')))
            #print("EW list : {}\n".format(par.contents.parContractsString[1].value.decode('utf-8')))
        par = par.contents.parScore[0].value.decode('utf-8')
        ns_score = par.split()[1]
        return int(ns_score)
    
        
    # Solutions
    #1	Find the maximum number of tricks for the side to play.  Return only one of the optimum cards and its score.
    #2	Find the maximum number of tricks for the side to play.  Return all optimum cards and their scores.
    #3	Return all cards that can be legally played, with their scores in descending order.

    def solve(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        results = self.solve_helper(strain_i, leader_i, current_trick, hands_pbn[:dds.MAXNOOFBOARDS], solutions)
        if len(hands_pbn) > dds.MAXNOOFBOARDS:
            i = dds.MAXNOOFBOARDS
            while i < len(hands_pbn):
                more_results = self.solve_helper(strain_i, leader_i, current_trick, hands_pbn[i:i+dds.MAXNOOFBOARDS], solutions)

                for card, values in more_results.items():
                    results[card] = results[card] + values

                i += dds.MAXNOOFBOARDS

        return results 

    def solve_helper(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        card_rank = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]

        self.bo.noOfBoards = min(dds.MAXNOOFBOARDS, len(hands_pbn))

        for handno in range(self.bo.noOfBoards):
            self.bo.deals[handno].trump = (strain_i - 1) % 5
            self.bo.deals[handno].first = leader_i

            for i in range(3):
                self.bo.deals[handno].currentTrickSuit[i] = 0
                self.bo.deals[handno].currentTrickRank[i] = 0
                if i < len(current_trick):
                    self.bo.deals[handno].currentTrickSuit[i] = current_trick[i] // 13
                    self.bo.deals[handno].currentTrickRank[i] = 14 - current_trick[i] % 13

            self.bo.deals[handno].remainCards = hands_pbn[handno].encode('utf-8')

            self.bo.target[handno] = -1
            # Return all cards that can be legally played, with their scores in descending order.
            self.bo.solutions[handno] = solutions
            self.bo.mode[handno] = self.dds_mode

        res = dds.SolveAllBoards(ctypes.pointer(self.bo), ctypes.pointer(self.solved))
        if res != 1:
            error_message = dds.get_error_message(res)
            print(f"{Fore.RED}Error Code: {res}, Error Message: {error_message} {hands_pbn[0].encode('utf-8')} {current_trick} {leader_i}{Style.RESET_ALL}")
            return None

        if solutions == 1:
            # Just return the maximum number of the side to play for each sample
            card_results = {}
            card_results["max"] = []
            card_results["min"] = []
            for handno in range(self.bo.noOfBoards):
                fut = ctypes.pointer(self.solved.solvedBoards[handno])
                suit_i = fut.contents.suit[0]
                card = suit_i * 13 + 14 - fut.contents.rank[0]
                card_results["max"].append(fut.contents.score[0])
                card_results["min"].append(fut.contents.score[fut.contents.cards-1])

        else:    
            card_results = {}
            for handno in range(self.bo.noOfBoards):
                fut = ctypes.pointer(self.solved.solvedBoards[handno])
                for i in range(fut.contents.cards):
                    suit_i = fut.contents.suit[i]
                    card = suit_i * 13 + 14 - fut.contents.rank[i]
                    if card not in card_results:
                        card_results[card] = []
                    card_results[card].append(fut.contents.score[i])
                    eq_cards_encoded = fut.contents.equals[i]
                    for k, rank_code in enumerate(card_rank):
                        if rank_code & eq_cards_encoded > 0:
                            eq_card = suit_i * 13 + k
                            if eq_card not in card_results:
                                card_results[eq_card] = []
                            card_results[eq_card].append(fut.contents.score[i])
        return card_results


    def expected_tricks_dds(self, card_results):
        return {card:round((sum(values)/len(values)),2) for card, values in card_results.items()}

    def expected_tricks_dds_probability(self, card_results, probabilities_list : List[float]):
        return {card: round(sum([p*res for p, res in zip(probabilities_list, result_list)]),2) for card, result_list in card_results.items()}

    def p_made_target(self, tricks_needed):

        def fun(card_results):
            return {card:round(sum(1 for x in values if x >= tricks_needed)/len(values),3) for card, values in card_results.items()}
        return fun

    def print_dd_results(self, dd_solved, print_result=True, xcards=False):
        print("DD Result\n".join(
            f"{Card.from_code(int(k))}: [{', '.join(f'{x:>2}' for x in v[:20])}{' ]' if len(v) <= 20 else ' ...]'}"
            for k, v in dd_solved.items()
        ))

        # Create a new dictionary to store sorted counts for each key
        sorted_counts_dict = {}

        # Loop through the dictionary and process each key-value pair
        for key, array in dd_solved.items():
            # Use Counter to count the occurrences of each element
            element_count = Counter(array)
            
            # Sort the counts by frequency in descending order
            sorted_counts = sorted(element_count.items(), key=lambda x: x[1], reverse=True)
            
            # Store the sorted result in the new dictionary
            sorted_counts_dict[key] = sorted_counts

        # Print the sorted counts for each key
        for key, sorted_counts in sorted_counts_dict.items():
            print(f"Sorted counts for {Card.from_code(int(key), xcards)} DD:")
            for value, count in sorted_counts:
                print(f"  Tricks: {value}, Count: {count}")
