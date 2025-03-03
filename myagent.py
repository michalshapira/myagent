"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import numpy as np
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState, SAONegotiator
from negmas import pareto_frontier


def aspiration_function(t, mx, rv, e):
    """A monotonically decreasing curve starting at mx (t=0) and ending at rv (t=1)"""
    # credit : https://www.yasserm.com/anl/tutorials/tutorial_develop/ (tutorial page)
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


def find_nearest_value_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class AwesomeNegotiator(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    def __init__(self, *args, e: float = 5.0, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        self.e = e
        # keeps track of our last estimate of the opponent reserved value
        self.opponent_rv = 1.0
        self.rational_outcomes = []
        self.pareto_outcomes = []
        self.step = 0
        self.relative_time = 0.0
        self.previous_offer = None

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.
        """
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            # enumerates outcome space when finite, samples when infinite
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > max(self.ufun.reserved_value, 0.80)
        ]

        # sort rational outcomes by their utility function values
        self.rational_outcomes.sort(
            key=lambda o: float(self.ufun(o)), reverse=True)

        pareto_utils, pareto_indices = pareto_frontier(
            (self.ufun, self.opponent_ufun), self.rational_outcomes)

        # get all pareto outcomes
        self.pareto_outcomes = [self.rational_outcomes[i]
                                for i in pareto_indices]

        # sort pareto outcomes by their utility function values
        self.pareto_outcomes.sort(
            key=lambda o: float(self.ufun(o)), reverse=True)

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.

        Remarks:
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        offer = state.current_offer
        self.relative_time = state.relative_time
        self.update_partner_reserved_value(state)

       # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun
        offer = state.current_offer

        # If there is no offer, there is nothing to accept
        if offer is None:
            return False

        if self.nmi.n_steps and self.nmi.n_steps - self.step == 1:
            # if in last step, accept any rational offer
            return self.ufun(offer) > self.ufun.reserved_value

        # Find the current aspiration level
        threshold = aspiration_function(
            self.relative_time, 1.0, self.ufun.reserved_value, self.e)

        self.step += 1

        return self.ufun(offer) > threshold

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """
        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.
        outcomes = self.pareto_outcomes if self.pareto_outcomes else self.rational_outcomes
        # print("pareto outcomes:---------------------------")
        # print(self.pareto_outcomes)

        # print("rational outcomes--------------------------")
        # print(self.rational_outcomes)

        # print("outcomes is pareto---------------------------------")
        # print(outcomes == self.pareto_outcomes)

        # print("options--------------------")
        # print(list(self.ufun.outcome_space.enumerate_or_sample()))

        if not outcomes:
            return self.ufun.best()

        if self.relative_time > 0.975 or (self.nmi.n_steps and self.nmi.n_steps - self.step == 1):
            # if it is the last step in the negotiation
            # - concede towards best option from best options of the opponent
            # inspired from UOagent in anl2024
            sorted_outcomes_for_opponent_best = sorted(
                outcomes, key=lambda o: float(self.opponent_ufun(o)), reverse=True)
            opponent_ufun_values = [float(self.opponent_ufun(
                o)) for o in sorted_outcomes_for_opponent_best]
            bot_idx = find_nearest_value_idx(
                opponent_ufun_values, self.opponent_rv + 0.02)
            top_idx = find_nearest_value_idx(
                opponent_ufun_values, self.opponent_rv)
            offers = sorted(
                sorted_outcomes_for_opponent_best[bot_idx:top_idx + 1], key=lambda o: float(self.ufun(o)), reverse=True)
            return offers[0] if offers else outcomes[0]

        asp_level = aspiration_function(
            state.relative_time, 1.0, 0.0, self.e)

        if self.previous_offer:
            # tit-for-tat
            if self.opponent_ufun(self.previous_offer) < self.opponent_ufun(state.current_offer):
                # concede
                asp_level *= 0.98
            else:
                # firm
                asp_level *= 1.01

        for outcome in outcomes:
            if self.ufun(outcome) >= asp_level:
                self.previous_offer = outcome
                return outcome

        return outcomes[0]

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        # assert self.ufun and self.opponent_ufun
        # offer = state.current_offer
        # if self.opponent_ufun(offer) < self.opponent_rv:
        #     self.opponent_rv = float(self.opponent_ufun(offer))

        # ########
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if (
            self.opponent_ufun(offer) < self.opponent_rv
            and self.opponent_ufun(offer) >= 0.01
        ):
            self.opponent_rv = float(self.opponent_ufun(offer))

        # update rational_outcomes by removing the outcomes that are below the reservation value of the opponent
        # Watch out: if the reserved value decreases, this will not add any outcomes.
        # rational_outcomes = self.rational_outcomes = [
        #     _
        #     for _ in self.rational_outcomes
        #     if self.opponent_ufun(_) > self.opponent_rv
        # ]


# # if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from .helpers.runner import run_a_tournament
    # from .helpers.runner_2 import anl2024_tournament

    run_a_tournament(AwesomeNegotiator, small=True, debug=True)
