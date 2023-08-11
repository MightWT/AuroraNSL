# This is a sample for converting hard coded logic files
from pyswip import Prolog
prolog = Prolog()
prolog.assertz("father(michael,john)")
prolog.assertz("father(michael,gina)")
print(list(prolog.query("father(michael,X)")) == [{'X': 'john'}, {'X': 'nick'}])
print(list(prolog.query("father(michael,nicle)")))
for soln in prolog.query("father(X,Y)"):
    print(soln["X"], "is the father of", soln["Y"])