"""Sensor-only data-driven future prediction. No ODE/Jacobian/cocycle imports here."""
import numpy as np

HORIZONS=np.array([0.2,0.5,1.0],float)
HISTORY=np.linspace(-0.5,0,17)
FEATURES=('instant','recent_0.10','history_0.50','per_trial_shuffled')
RIDGES=np.array([1.e-7,1.e-5,1.e-4,1.e-3,1.e-2,.1,1.,10.])


def feature_view(X,mode,permutations=None):
    """X: [trajectories,17 times, observable dim]. Endpoint must be last."""
    x=np.asarray(X,float)
    if mode=='instant':return x[:,-1,:].copy()
    if mode=='recent_0.10':return x[:,-4:,:].reshape(len(x),-1)
    if mode=='history_0.50':return x.reshape(len(x),-1)
    if mode=='per_trial_shuffled':
        if permutations is None:raise ValueError('Permutation control requires trajectory-specific permutation')
        return np.take_along_axis(x,np.asarray(permutations)[:,:,None],axis=1).reshape(len(x),-1)
    raise ValueError(mode)


def scrambled_indices(n,seed):
    rng=np.random.default_rng(seed)
    return np.array([rng.permutation(len(HISTORY)) for _ in range(n)])


def ridge_fit(X,Y,lam):
    """Standardize on train data only. Fitted predictor has no physical state labels."""
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    mean=X.mean(axis=0);scale=np.maximum(X.std(axis=0),1e-7)
    XX=(X-mean)/scale;target=Y.mean(axis=0)
    U,s,Vt=np.linalg.svd(XX,full_matrices=False)
    coef=Vt.T@((s/(s*s+lam))[:,None]*(U.T@(Y-target)))
    return (mean,scale,target,coef)


def ridge_predict(model,X):
    mean,scale,target,coef=model
    return ((np.asarray(X)-mean)/scale)@coef+target


def rmse(pred,truth):
    return float(np.sqrt(np.mean((np.asarray(pred)-np.asarray(truth))**2)))


def evaluate_blind(trainX,trainY,valX,valY,testX,testY,noise_sigma,seed=1823):
    """Fit only observed trajectories and observed futures; no full state labels."""
    rng=np.random.default_rng(seed)
    # Independent observation noise in each trajectory and time/channel.
    noises=[rng.normal(0,noise_sigma,size=x.shape) for x in (trainX,valX,testX)]
    xtr=trainX+noises[0];xva=valX+noises[1];xte=testX+noises[2]
    outputs={}; predictions={}; models={}
    dims=(trainX.shape[-1],len(HISTORY)*trainX.shape[-1])
    for i,mode in enumerate(FEATURES):
        if mode=='per_trial_shuffled':
            ptr=scrambled_indices(len(xtr),seed+101)
            pva=scrambled_indices(len(xva),seed+102)
            pte=scrambled_indices(len(xte),seed+103)
        else:ptr=pva=pte=None
        tr=feature_view(xtr,mode,ptr);va=feature_view(xva,mode,pva);te=feature_view(xte,mode,pte)
        optimum=None
        for lam in RIDGES:
            model=ridge_fit(tr,trainY,lam)
            score=rmse(ridge_predict(model,va),valY)
            if optimum is None or score<optimum[0]:optimum=(score,float(lam),model)
        _,lam,_=optimum
        # After selection, train on train ONLY (validation never used for coefficients).
        model=ridge_fit(tr,trainY,lam)
        pred=ridge_predict(model,te)
        outputs[mode]={'validation_rmse':optimum[0],'test_rmse':rmse(pred,testY),
                       'ridge_lambda':lam,'feature_count':tr.shape[1]}
        predictions[mode]=pred;models[mode]=model
    # SVD of observed histories: dimensionality diagnostic only; no hidden labels.
    centered=feature_view(xtr,'history_0.50')-feature_view(xtr,'history_0.50').mean(axis=0)
    spectrum=np.linalg.svd(centered,compute_uv=False)
    outputs['history_singular_values']=spectrum[:min(15,len(spectrum))].tolist()
    outputs['sensor_only_training']=True
    outputs['test_baseline_rmse']=rmse(np.repeat(trainY.mean(axis=0,keepdims=True),len(testY),axis=0),testY)
    return outputs,predictions


def delay_latent_rank_scan(trainX,trainY,valX,valY,testX,testY,noise_sigma,seed=823):
    """Unsupervised SVD sensor-history coordinates; learned predictive closure.

    No physical state or ODE enters the latent model. Latent coordinates are
    identifiable only up to an invertible gauge, not the true material state.
    """
    rng=np.random.default_rng(seed)
    tr=feature_view(trainX+rng.normal(0,noise_sigma,size=trainX.shape),'history_0.50')
    va=feature_view(valX+rng.normal(0,noise_sigma,size=valX.shape),'history_0.50')
    te=feature_view(testX+rng.normal(0,noise_sigma,size=testX.shape),'history_0.50')
    mean=tr.mean(axis=0);scale=np.maximum(tr.std(axis=0),1.e-7)
    zz=(tr-mean)/scale
    U,s,Vt=np.linalg.svd(zz,full_matrices=False)
    results=[];candidate=(3,4,6,8,10,12,15,20)
    for k in candidate:
        if k>len(s):continue
        V=Vt[:k].T
        F=zz@V
        G=((va-mean)/scale)@V
        Z=((te-mean)/scale)@V
        beta=np.linalg.lstsq(np.c_[np.ones(len(F)),F],trainY,rcond=None)[0]
        pv=np.c_[np.ones(len(G)),G]@beta
        pt=np.c_[np.ones(len(Z)),Z]@beta
        results.append(dict(rank=k,validation_rmse=rmse(pv,valY),test_rmse=rmse(pt,testY),
             explained_train_variance=float(np.sum(s[:k]**2)/np.sum(s**2))))
    winner=choose_predictive_latent_rank(results,valY)
    return results,winner


def choose_predictive_latent_rank(results,validation_targets):
    """Validation-only model choice with an explicit floating-point tie rule.

    If differences in validation error are at roundoff level, choose the
    smallest latent dimension. This is a parsimony rule, NOT a physical-state
    dimension assumption. Larger dimensions remain eligible when they offer
    a meaningful improvement on validation data.
    """
    if not results:
        raise ValueError('No candidate latent models')
    target=np.asarray(validation_targets,dtype=float)
    if not np.all(np.isfinite(target)):
        raise ValueError('Non-finite validation targets')
    best_error=min(float(row['validation_rmse']) for row in results)
    # Dimensionless relative to typical future-target magnitude; includes an
    # absolute 1e-12 floor for exact synthetic data at machine precision.
    target_rms=float(np.sqrt(np.mean(target**2)))
    tie_tolerance=1e-12*max(1.0,target_rms)
    near=[row for row in results
          if float(row['validation_rmse'])<=best_error+tie_tolerance]
    selected=min(near,key=lambda row:(int(row['rank']),float(row['validation_rmse'])))
    return {**selected,
            'validation_best_rmse':best_error,
            'validation_tie_tolerance':tie_tolerance,
            'rank_selection_rule':'lowest_rank_within_validation_numerical_tie',
            'physical_state_dimension_inferred':False}
