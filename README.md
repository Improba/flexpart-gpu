# FLEXPART-GPU

Port Rust/WebGPU de FLEXPART pour exploration scientifique, validation numerique et evaluation des performances CPU (Fortran/Rust) vs GPU.

## Contexte

Ce projet re-implemente des composants de FLEXPART en Rust et WGSL, avec un focus sur:

- la reproductibilite des experiences,
- la comparaison de sorties entre implementation historique et port GPU,
- la mesure de performances sur des tailles de problemes significatives.

## Licence et implications

FLEXPART amont est distribue sous licence `GPL-3.0-or-later`.
Le present projet est donc publie sous `GPL-3.0-or-later` egalement.

Implications pratiques pour la distribution:

- ce port doit rester sous licence compatible GPL;
- si des binaires sont distribues, le code source correspondant doit etre fourni;
- les notices de copyright/licence et attributions amont doivent etre conservees;
- les modifications doivent etre clairement identifiees;
- l'assistance IA au portage n'annule pas les obligations de la licence.

## Attribution

Ce projet reconnait explicitement l'origine scientifique et logicielle de FLEXPART.
Le credit principal des fondations modeles, des methodes et de la validation scientifique revient a l'equipe FLEXPART.

Voir `NOTICE.md` pour les details d'attribution et de conformite.

## Benchmarks

La methode de benchmark et les commandes recommandees sont documentees dans `docs/benchmarks.md`.

## Contact

Sylvain Meylan (Improba)  
Email: `A_RENSEIGNER`
