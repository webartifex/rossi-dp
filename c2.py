"""Chapter 2: Optimal Stock Market Strategy"""

import functools
import math
import random

from typing import Optional, Sequence


class MaxProfit:
    """Find the max. profit with a simple stock trading strategy.

    Given a sequence of `float`s resembling the daily prices of a stock,
    the task is to find the best possible trading strategy.
    To simplify, there is only one "average" price per day.

    At the end of each day, the trader either owns or does not own a single stock.

    So, there are four possible transitions:

    - If one does NOT own the stock at the end of the previous day:
      1. BUY the stock, or
      2. DO NOT BUY the stock.

    - If one owns the stock at the end of the previous day:
      3. SELL the stock, or
      4. HOLD the stock.
    """

    def __init__(
        self, prices: Sequence[float], initial_budget: Optional[float] = None
    ) -> None:
        """Initialize the algorithm.

        The problem can be solved with or without allowing to borrow money.

        Args:
            prices: daily stock prices
            initial_budget: trader's net worth in the beginning;
                if set, the net worth may never become negative throughout
                the time horizon meaning that the trader may NOT borrow money;
                must be non-negative if provided
        """
        self._prices = list(float(x) for x in prices)

        # Assume an "unlimited" `initial_budget` if it is set to `None`.
        if initial_budget is None:
            # "Unlimited" means `sum(prices) + 1`, rounded up.
            # Such an `._initial_budget` suffices to make all possible "buy"s.
            self._initial_budget = math.ceil(sum(self._prices) + 1)
        else:
            self._initial_budget = float(initial_budget)

    def solve(self, bottom_up: bool = True) -> float:
        """Solve the problem.

        Args:
            bottom_up: defaults to using a "bottom-up" strategy;
                otherwise, a "top-down" (= recursive) strategy can be used

        Returns:
            max_profit
        """
        if bottom_up:
            return self._bottom_up()

        # Must subtract `._initial_budget` as that is the return value in the
        # base case and we are interested in the profit and not the net worth.
        last_day = len(self._prices) - 1
        return self._top_down(last_day, has_stock=False) - self._initial_budget

    def _bottom_up(self) -> float:
        """Looping strategy to find the max. profit.

        Returns:
            max_profit
        """
        # At each time step, we need to track two separate states:
        #  1) either the trader already owns the stock, or
        #  2) he does not.
        # That is modeled with variables `cash_with_stock` and `cash_without_stock`
        # that track the cash the trader has before a transition is executed.
        # To initialize the variables, we set `cash_with_stock = -float("inf")`
        # as the trader by assumption does not own the stock in the beginning.
        # Setting `cash_without_stock = 0` would normalize the cash amount such
        # that its value at the end is the profit (or loss). So, `cash_without_stock`
        # may be negative at some intermediate time steps. Then, the trader is
        # borrowing money. To generalize this method such that it may be used
        # with borrowing not allowed, we set `cash_without_stock` to
        # `._initial_budget` instead of `0`.
        cash_with_stock = -float("inf")
        cash_without_stock = self._initial_budget

        for price in self._prices:
            # 1) The trader owns the stock at the end of the day.
            buy = cash_without_stock - price
            hold = cash_with_stock

            # 2) The trader does NOT own the stock at the end of the day.
            do_not_buy = cash_without_stock
            sold = cash_with_stock + price

            # Choose the better action at the end of the day.
            cash_with_stock = max(buy, hold)
            cash_without_stock = max(do_not_buy, sold)

            # To not allow borrowing, we choose a big penalty that is never optimal.
            if cash_with_stock < 0:
                cash_with_stock = -float("inf")

        # By assumption, the trader sells the stock on the last day.
        # The `._initial_budget` is not part of the profit.
        return cash_without_stock - self._initial_budget

    @functools.lru_cache(maxsize=None)
    def _top_down(self, day: int, has_stock: bool) -> float:
        """Recursive strategy to find the max. profit at the end of a `day`.

        Args:
            day: 0-based index of the day for which the max. profit is calculated;
                must be non-negative as otherwise the base case is not reached
            has_stock: if the trader owns the stock at the end of the `day`

        Returns:
            max_profit
        """
        # Base case => the end of the day BEFORE the day of the first price in
        # `._prices`. By assumption, the trader may NOT own the stock then.
        # That is modeled with a high penalty (i.e., `-float("inf")`) so that it
        # is not optimal to be chosen. Otherwise, there is no profit yet. In that
        # case, we return `._initial_budget`, which must be subtracted again in the
        # `.solve()` method above. Again, this way, this method may model the case
        # where borrowing is not allowed as well.
        if day < 0:
            if has_stock:
                return -float("inf")
            return self._initial_budget

        price = self._prices[day]

        if has_stock:
            # If the trader owns the stock at the end of the `day`, he either
            # has to buy it or has had it on the previous day already.
            buy = self._top_down(day - 1, has_stock=False) - price
            hold = self._top_down(day - 1, has_stock=True)

            # To prevent borrowing, we return `-float("inf")` as a penalty if the
            # net worth becomes negative. As `._initial_budget` is the start value,
            # `net_worth` becomes negative only if the cumulative trading losses
            # are larger than the initial budget. Again, the condition can only be
            # `True` if the class is instantiated with an `initial_budget`.
            if (net_worth := max(buy, hold)) < 0:
                return -float("inf")
            return net_worth

        # If the trader does NOT own the stock at the end of the `day`, he
        # either has not bought it on the `day` or sold it.
        do_not_buy = self._top_down(day - 1, has_stock=False)
        sell = self._top_down(day - 1, has_stock=True) + price
        return max(do_not_buy, sell)


# Test cases provided by the book.

example1 = MaxProfit([2, 5, 1])

result1_bu = example1.solve(bottom_up=True)
result1_td = example1.solve(bottom_up=False)

assert result1_bu == result1_td == 3


example2 = MaxProfit([2, 5, 1, 3])

result2_bu = example2.solve(bottom_up=True)
result2_td = example2.solve(bottom_up=False)

assert result2_bu == result2_td == 5


# Randomly generated test cases: both "bottom-up" and "top-down"
# approaches must provide the same answer. Otherwise, something is broken.
for n_days in range(10, 101):
    prices = [random.randint(1, 50) for _ in range(n_days)]

    example = MaxProfit(prices)
    assert example.solve(bottom_up=True) == example.solve(bottom_up=False)

    no_borrowing = MaxProfit(prices, initial_budget=10)
    assert no_borrowing.solve(bottom_up=True) == no_borrowing.solve(bottom_up=False)
