"""Geometry of the residual--description-length frontier, with fixed scales."""
import numpy as np

def knee(front,numerical_floor=1e-12):
    """Choose maximum error reduction per normalized description-length gain.

    This is the maximum signed distance to the endpoint chord of the log-error
    Pareto curve.  It reads only observed validation error and expression size.
    Equivalent errors below the integration floor use the shortest expression.
    """
    ordered=sorted(front,key=lambda r:r['complexity'])
    if not ordered:raise ValueError('Empty Pareto frontier')
    useful=[];smallest=float('inf')
    for row in ordered:
        error=max(row['validation_relative_mse'],numerical_floor)
        if error<smallest:
            useful.append(row);smallest=error
    if len(useful)<3:return useful[-1],{'rule':'normalized_log_frontier_knee','frontier_points':len(useful)}
    x=np.array([r['complexity'] for r in useful],float)
    logerr=np.log10(np.maximum([r['validation_relative_mse'] for r in useful],numerical_floor))
    x=(x-x[0])/max(x[-1]-x[0],1)
    improvement=(logerr[0]-logerr)/max(logerr[0]-logerr[-1],1e-30)
    distances=improvement-x
    index=int(np.argmax(distances)) if np.max(distances)>1e-10 else len(useful)-1
    return useful[index],{'rule':'normalized_log_frontier_knee','frontier_points':len(useful),
                          'numerical_floor':numerical_floor,'selected_distance':float(distances[index]),
                          'log_error_span':float(logerr[0]-logerr[-1])}
