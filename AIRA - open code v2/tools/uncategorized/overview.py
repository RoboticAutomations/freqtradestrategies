"""
.. module:: group.overview
   :synopsis: group overview table.

.. moduleauthor:: Tianning Li <ltianningli@gmail.com>
"""

from src.finvizfinance.group.base import Base


class Overview(Base):
    """Overview
    Getting information from the finviz group overview page.
    """

    v_page = 110
