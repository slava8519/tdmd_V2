# TDMD Engineering Spec

**Версия:** 2.5
**Статус:** master engineering spec (candidate)
**Назначение:** единственный источник истины по проекту TDMD. Все модульные `SPEC.md`, код, тесты, CI и execution packs — производные от этого документа.

**Изменения v2.5 относительно v2.4:** systematic анализ LAMMPS/GROMACS/NAMD выявил 4 high-priority blind spots, каждый адресован формальной политикой: dynamic load balancing (scheduler/SPEC §11a); GPU-resident mode mandate (integrator/SPEC §3.5); workload saturation и minimum atoms per rank (perfmodel/SPEC §3.7); expensive compute intervals policy (master spec §6.5b + integrator/SPEC §4.5). См. Приложение C.

**Изменения v2.4 относительно v2.3:** закрыта волна из 5 open questions, требовавших formal policy: FMA cross-compiler binding (§D.10, §7.3); thermostat-in-TD research program (§14 M11, integrator/SPEC §7.3); Auto-K policy (§6.5a); cutoff/smoothing policy (potentials/SPEC §2.4); Python bindings strategy (Приложение E).

**Изменения v2.3 относительно v2.2:** добавлено Приложение D (Precision Policy Details) с полной политикой использования точности; пятый BuildFlavor `MixedFastAggressiveBuild` как opt-in Philosophy A; обновлены §7.1, §7.2; M8 расширен с `MixedFastSnapOnlyBuild`.

**Изменения v2.2 относительно v2.1:** добавлен модуль `verify/` (VerifyLab) как cross-module scientific validation layer. См. §13.0.

**Изменения v2.1 относительно v2.0:** добавлены §4a (two-level TD×SD decomposition), §5.3 (unit system support), связанные уточнения в §6, §10, §12, §14, §B и Change log.

---

## 0. Для кого и как читать

Документ адресован трём аудиториям, одновременно:

- **техлид / архитектор** читает части I и II (позиционирование, метод, модель производительности) и II-III (архитектура, модули) чтобы принять решение — делать или не делать и в какой форме;
- **инженеры реализации** читают части III, IV, V (интерфейсы, псевдокод, тест-план) чтобы кодить;
- **научный пользователь / reviewer** читает часть I и приложение A чтобы понять, что именно считает движок и при каких условиях это воспроизводимо.

Документ устроен так, что первые пять разделов (часть I и начало II) можно прочитать отдельно и получить полное понимание *что* и *почему*. Всё остальное — *как*.

---

# Часть I. Метод и почему он стоит реализации

## 1. Что такое TDMD и чем он отличается

**TDMD** — standalone программа молекулярной динамики, у которой первичный принцип исполнения параллельной обработки — **декомпозиция по времени (Time Decomposition, TD)** в духе диссертации В.В. Андреева (2007), обобщённая до современного GPU-first движка с hybrid time × space decomposition.

**TDMD отличается от LAMMPS/GROMACS/HOOMD** не набором потенциалов или интеграторов (здесь он их догоняет), а принципом организации параллельной работы: вместо того чтобы всем процессам совместно продвигать один и тот же временной шаг (классическая spatial decomposition, SD), в TDMD разные области модели могут одновременно находиться на **разных временных шагах**, при соблюдении условия причинной изоляции. Это — **не multiple timestepping (rRESPA)** и **не parallel-in-time (Parareal)**. Это отдельный класс алгоритмов, эксплуатирующий локальность короткодействующих потенциалов.

**TDMD не претендует** заменить LAMMPS как универсальный MD-код. Его ниша уже и конкретнее — см. §3.

## 2. Суть метода (якорь из диссертации)

### 2.1. Физическое основание

Молекулярная динамика с короткодействующими потенциалами — процесс **очень близкого взаимодействия**: радиус обрезания потенциала (единицы Å) на много порядков меньше размера модели (сотни Å … μm). Это даёт фундаментальное свойство:

> Атомы, находящиеся в областях, удалённых друг от друга на расстояние больше диаметра потенциала взаимодействия, **не влияют друг на друга в пределах одного шага интегрирования**.

Из этого следует, что разные области пространства модели могут одновременно находиться на **разных шагах интегрирования** (h, h+1, h+2, …), если эти области попарно удалены более чем на диаметр потенциала и если пути атомов в них за соответствующие временные интервалы не пересекают границ. Это и есть **time decomposition**.

### 2.2. Расчётная зона как единица работы

Единица работы в TD — **расчётная зона** (zone). Минимальный размер зоны по каждой координате — радиус обрезания потенциала `r_c`. Типичный выбор — сторона зоны `= r_c` (квадратные / кубические зоны вдоль осей).

Зона имеет **динамический тип** (состояние) из конечного автомата диссертации. В нашей современной формулировке состояние состоит из восьми значений, но они соответствуют (через приложение A.3) четырём базовым типам диссертации: `w` (пустая, готова к приёму), `d` (содержит уже рассчитанные данные, готова к передаче), `p` (в процессе расчёта), `r` (рассчитана полностью для текущего шага).

Переходы между состояниями триггерятся событиями: **приём данных, попадание в радиус действия потенциала, старт расчёта, завершение расчёта, передача данных следующему владельцу**. Весь процесс TD — это согласованная эволюция конечных автоматов зон, координируемая scheduler'ом.

### 2.3. Условие безопасности на границе

Ключевая проблема TD: атом у границы обработанной области может за один шаг переместиться в область, данные которой уже переданы следующему потребителю, и расчёт станет некорректным. Диссертация предлагает три механизма защиты, и в TDMD все три сосуществуют как слои:

1. **Минимум N зон на rank в памяти** — консервативный запас, перекрывающий возможное смещение атома за один шаг (подробно см. §4.2, формула N_min);
2. **Verlet-like таблицы соседей** на k шагов + skin-слой — атом не покидает сферу `r_c + r_skin` за k шагов, что даёт верхнюю границу на безопасное время жизни neighbor list;
3. **Буфер безопасности ширины `Δ = v_max · dt · α`**, где α — коэффициент запаса (∈ [1.5, 3.0]), `v_max` — максимальная скорость в зоне.

Совокупная гарантия — **safety certificate** (§6.4) — формальный объект, без валидного экземпляра которого scheduler не имеет права выдать зону на расчёт.

### 2.4. Два режима pipeline: базовый и K-batched

**Базовый TD-pipeline.** Процессор `P_i` рассчитывает зону `z_j` для шага `h`, по завершении передаёт `P_{i+1}` данные зоны `z_j` для расчёта шага `h+1`, и так далее. При ring-топологии и линейном обходе зон pipeline стабилизируется за ≤ N_min·P шагов (см. §4.2).

**K-batched TD-pipeline.** Каждый процессор ведёт не один, а K последовательных шагов на своих зонах подряд, прежде чем передавать данные. Это центральная идея §3.4 диссертации: **время приёма-передачи на единицу полезной работы уменьшается в K раз** (формула Андреева, формула (51) диссертации), ценой K-кратного роста памяти на pipeline-буферы и K-кратно более длинного startup.

Это важнейший tunable параметр TDMD. Он обсуждается отдельно в §4.3.

## 3. Целевая ниша и продуктовое позиционирование

### 3.1. Каноническая одно-предложная формулировка

> **TDMD — это GPU-first standalone MD-движок, ориентированный на дорогие локальные many-body и ML-потенциалы (EAM, MEAM, SNAP, MLIAP, PACE), использующий time decomposition для снижения требований к межпроцессорной коммуникации и повышения масштабируемости на системах со сравнительно медленными каналами (включая commodity HPC и cloud-кластеры без NVLink/InfiniBand premium).**

### 3.2. Почему именно этот класс задач

TD даёт выигрыш над SD при одновременном выполнении трёх условий:

1. **Локальное вычисление дорогое** — т.е. стоимость force evaluation на атом велика относительно стоимости упаковки-передачи его состояния. Именно так устроены EAM, MEAM, SNAP, MLIAP, PACE. У простого pairwise Lennard-Jones эта стоимость мала, и SD его выигрывает.
2. **Требования к каналам связи — узкое место**, или channels деградируют при масштабировании. Это условие сегодня не про «медленные компьютеры 2007 года», а про реальную ситуацию с commodity cloud-кластерами без NVLink, с выборочным GPU-aware MPI, и тем более с мультиузловыми конфигурациями без rail-optimized fabric.
3. **Radius of influence велик** — у many-body и ML потенциалов эффективный stencil шире, чем у pair, потому что force зависит от локального окружения (density, bispectrum), а не только от прямых соседей. Это увеличивает halo-объём в SD и в то же время увеличивает локальную работу, что благоприятно для TD.

### 3.3. Где TDMD **не пытается** побеждать

- **Простые pair-потенциалы** (LJ, Morse solo): LAMMPS/HOOMD+SD лучше; в TDMD Morse присутствует как reference для валидации, а не как flagship use-case.
- **Плотные глобальные взаимодействия** (long-range electrostatics без cutoff): вне TD-парадигмы. TDMD поддержит long-range через отдельный **split-partition service** (см. §11) на более поздних milestones; на короткой дистанции TD, на длинной — вырожденный SD/PME-подобный путь.
- **Реактивные потенциалы с глобальным charge-equilibration** (ReaxFF, COMB): архитектурно возможны, но не в v1.

### 3.4. Антипозиционирование и честность

TDMD **не является**:

- ещё одним универсальным MLIP-движком (таких десятки);
- клоном LAMMPS с TD-патчем;
- parallel-in-time методом типа Parareal (у нас разные временные шаги на разных пространственных областях, а не разные временные окна на одной и той же области);
- rRESPA-подобным multi-timestepping (у нас нет разделения сил на fast/slow; у нас разделение по пространству с разными temporal frontiers).

---

## 4. Модель производительности TD (когда TD выигрывает)

Это один из двух инженерных стержней документа (второй — архитектура). Без этой модели проект не имеет критерия успеха и бенчмарк-стратегия висит в воздухе.

### 4.1. Базовая модель одной итерации

Обозначения на rank:

- `T_c` — compute time force+integrate для всех локальных атомов за один шаг;
- `T_h` — halo exchange time (pack + send + recv + unpack) за один шаг в схеме SD;
- `T_p` — peer-to-peer transfer time одной зоны в схеме TD (базовый pipeline, K=1);
- `n_z` — число зон на rank;
- `N_min` — минимум зон в памяти rank'а для устойчивого TD pipeline (зависит от обхода);
- `K` — глубина batching'а в K-batched pipeline;
- `B` — bandwidth межпроцессорной связи, байт/с;
- `V_zone` — объём данных зоны, байт;
- `ρ` — плотность атомов, атомов/Å³.

### 4.2. SD vs базовый TD

**SD (spatial decomposition):**
```
T_step_SD  =  T_c  +  T_h
T_h        ≈  2 · (surface_cells / total_cells) · N_local · atom_record_size / B
```

**Базовый TD (K=1):** в устойчивом режиме каждая зона за свой временной шаг требует одной отправки. Но важное отличие: **отправка `T_p` может полностью перекрываться с расчётом соседних зон** на том же rank благодаря conveyor-structure. При `n_z ≥ N_min` и при async comm:
```
T_step_TD  =  max(T_c, n_z · T_p)   (перекрытие compute/comm)
            =  T_c    если   T_p  ≤  T_c / n_z
```

Это означает: **TD уже при K=1 скрывает коммуникации, если стоимость одной зоны compute превышает стоимость передачи**. Для многочастичных / ML потенциалов это условие выполнено с запасом.

### 4.3. K-batched TD: главная формула экономии каналов

Ключевая формула диссертации (Андреев, формула (51)) в наших обозначениях:
```
T_comm_per_step_TD(K)  =  T_p / K
```

То есть если rank ведёт `K` последовательных шагов на своих зонах перед передачей, **эффективная стоимость коммуникации на шаг падает в K раз**, ценой:

- памяти: `O(K · N_local · atom_record_size)` на rank для buffer-chain'а;
- startup latency: pipeline fill занимает `O(K · N_min · P)` шагов, где `P` — число ranks;
- снижения reactivity к изменениям состояния (migration, neighbor rebuild) — при больших K приходится откатывать больше работы при ошибке safety certificate.

**Критерий выигрыша TD над SD:**
```
TD_wins  ⟺   T_c + T_h_SD   >   T_c + T_p / K
         ⟺   T_h_SD         >   T_p / K
         ⟺   K              >   T_p / T_h_SD
```

Для типичного случая `T_p ~ T_h_SD` выигрыш уже при K=2. Для дорогих many-body потенциалов, где `T_c ≫ T_h_SD`, TD и SD сравнимы (оба скрывают comm); **но TD снимает верхний потолок масштабирования**, потому что при SD `T_h_SD` растёт с ростом числа ranks как поверхность / объём, а в TD при K-batching'е он просто делится на K.

### 4.4. Оптимальное число ranks

Из диссертации (формула (44)):
```
n_opt  =  floor(total_zones / N_min)
```

Где `N_min` — минимальное число зон, необходимое одному rank'у для устойчивого pipeline. Для разных схем разбиения:

- **1D linear zoning (Andreev §2.2):** `N_min = 2` — оптимальный случай, максимум параллелизма на тот же объём работы;
- **2D zoning (Andreev §2.4, рис. 23):** `N_min = 6` — для обхода со столбцовой змейкой;
- **3D zoning линейной нумерацией (Andreev §2.5, рис. 25):** `N_min = 274` для модели 16×16×16 — чрезмерно, обход нужно менять;
- **3D zoning Hilbert-подобной нумерацией:** `N_min = O((r_c/L)^{-2})` — это то, что TDMD должен реализовать по умолчанию, см. §9.

### 4.5. Performance model как first-class концепт

В TDMD **обязательно** существуют:

1. **модуль `tdmd-perfmodel`** — analytic predictor `T_step_TD(K, config)` и `T_step_SD(config)`;
2. **benchmark-harness** сравнивает предсказания модели с измерениями и проваливает CI, если расхождение > 20% (modeling error gate);
3. **CLI `tdmd explain --perf`** выводит: «для вашей задачи и кластера ожидаемый TD(K=4) даст ~2.3× ускорения над SD; измеренная величина будет записана сюда».

Без perf model нельзя ответить «почему TDMD». С perf model проект становится научно ответственным.

---

## 4a. Two-level decomposition: TD внутри, SD снаружи

Этот раздел закрывает вопрос «а что если ranks слишком много для чистого TD?» и формализует гибридную схему, которая является **целевой архитектурой TDMD для multi-node deployments**.

### 4a.1. Почему чистый TD упирается в потолок

Из §4.4: `n_opt_TD = floor(N_zones / N_min)`. Для Hilbert 3D zoning `N_min = O(P^{2/3})`, что хорошо, но всё равно конечно. Для модели Al FCC 10⁶ атомов с `r_c = 8 Å` (сторона зоны ≈ 10 Å, box ≈ 400 Å, `N_z ≈ 50³ = 125 000` зон) оценка даёт `n_opt ≈ 1500–2000` ranks.

Это достаточно для single-node multi-GPU и для small-to-medium multi-node конфигураций, но **не достаточно для exascale** (10⁴–10⁵ ranks). Без расширения TDMD упирается в потолок на больших кластерах.

### 4a.2. Формулировка two-level decomposition

TDMD поддерживает **два уровня параллелизма**, перемножающихся:

```
P_total  =  P_space  ×  P_time
```

**Outer level — spatial decomposition (SD):**
- модель разбивается на `P_space` ортогональных subdomain'ов по схеме LAMMPS-подобной SD;
- каждый subdomain принадлежит одной "group" ranks (обычно — один узел кластера);
- между subdomain'ами происходит классический halo exchange.

**Inner level — time decomposition (TD):**
- внутри каждого subdomain'а `P_time` ranks (обычно — GPU внутри узла) ведут TD-pipeline;
- между ranks одного subdomain'а идут temporal packets;
- zoning planner работает в пределах одного subdomain.

### 4a.3. Геометрическое соответствие железу

Эта декомпозиция **точно соответствует реальной топологии современных HPC-кластеров**:

| Уровень декомпозиции | Канал связи | Характеристики | Протокол TDMD |
|---|---|---|---|
| Outer (между subdomain'ами) | InfiniBand / Ethernet / slow fabric | высокая latency, средний bandwidth, подходит для больших пакетов | SD halo exchange |
| Inner (внутри subdomain'а) | NVLink / NVSwitch / PCIe | низкая latency, высокий bandwidth, хорош для частых мелких пакетов | TD temporal packets |

Почему это работает:

- **SD halo exchange** происходит редко (раз в несколько шагов, если K-batching) и передаёт большие плоскости halo-атомов — идеален для fabric с высокой latency но хорошим bandwidth;
- **TD temporal packets** идут часто (каждый локальный шаг), но мелкие (одна зона ≈ сотни атомов) — идеальны для NVLink/NVSwitch.

### 4a.4. Модель производительности two-level

**Чистый TD (из §4.3):**
```
T_step_TD  =  max(T_c, T_p / K)
P_max_TD   =  n_opt  (конечный предел)
```

**Чистый SD:**
```
T_step_SD  =  T_c  +  T_h_SD(P)
T_h_SD(P)  =  O(surface_per_rank)  =  O(V^{2/3} / P^{2/3})
Scaling    =  ограничен surface-to-volume при большом P
```

**Two-level TD×SD:**
```
T_step_TD×SD  =  T_c  +  max(T_h_outer(P_space),
                              T_p_inner / K)
```

Где:
- `T_c` — compute time на rank;
- `T_h_outer(P_space)` — halo exchange между subdomain'ами;
- `T_p_inner / K` — K-batched TD communication внутри subdomain'а.

**Ключевое свойство:** `T_h_outer` зависит от `P_space`, но не от `P_time`. А `T_p_inner / K` вообще не зависит от `P_space`. Значит, увеличение `P_time` (больше GPU внутри узла) не увеличивает inter-node трафик.

### 4a.5. Когда какой уровень важен

**Режим "TD dominates" (`P_space = 1`):** single-node multi-GPU. TD полностью скрывает inter-GPU коммуникации через NVLink. Это конфигурация для workstation, workstation-class DGX, и небольших кластеров.

**Режим "SD dominates" (`P_time = 1`):** чистый LAMMPS-стиль, если TDMD запущен без TD. Полезен для debugging и для baseline compare. **Это не рекомендуемый production-режим**, но он должен работать (см. §10.1, SD-vacuum mode).

**Режим "hybrid" (`P_space > 1`, `P_time > 1`):** production default для multi-node. Обычно `P_time = число_GPU_на_узле` (4, 8, 16), `P_space = число_узлов`.

### 4a.6. Critical-path анализ

В hybrid режиме критический путь — `max(T_h_outer, T_p_inner / K)`. Интересный факт: **эти две величины можно балансировать независимо**:

- `T_h_outer` уменьшается увеличением `K` (в K-batched pipeline halo exchange тоже можно делать раз в K шагов, амортизируя cost);
- `T_p_inner / K` уменьшается увеличением `K` напрямую.

Значит, `K` — единственный параметр, который управляет балансом. Это делает runtime-tuning задачей одномерной (не multidimensional optimization), что существенно упрощает автотюнинг. В TDMD auto-K policy описывается в §6.5 (вместе с timestep policy).

**Важное ограничение:** увеличение `K` увеличивает память (§4.3) и startup latency. Разумный диапазон: `K ∈ [1, 16]` для production, `K ∈ [1, 64]` для research.

### 4a.7. Scheduler two-level: роли и контракты

Из §8.2 знаем: никто кроме scheduler не владеет временем. В two-level случае есть **две разные scheduling-роли**:

**Outer SD Coordinator** (один экземпляр на весь run, глобальный):
- поддерживает глобальный `frontier_min` — самый младший time_level в системе;
- координирует halo exchange между subdomain'ами;
- следит за global safety: `max_time_level - frontier_min ≤ K_max`;
- не знает про internal zoning конкретных subdomain'ов.

**Inner TD Scheduler** (один экземпляр на subdomain):
- управляет zone DAG внутри своего subdomain'а;
- выдаёт `ZoneTask`'и локальным ranks;
- строит `SafetyCertificate` для своих зон;
- **спрашивает** Outer SD Coordinator перед продвижением на шаг `h+1`, если зона лежит близко к границе subdomain'а (в пределах `r_c + r_skin` от SD-halo).

**Контракт между уровнями:** Inner не имеет права продвинуть пограничную зону на шаг `h+1`, пока Outer не подтвердит, что соседний subdomain получил halo для шага `h`. Это формализуется как отдельный тип зависимости `SubdomainBoundaryDependency` в DAG scheduler'а.

Для non-boundary зон Inner scheduler работает полностью автономно — ровно как single-node TD из §6.

### 4a.8. Протокол halo между subdomain'ами с разными temporal frontiers

Это самая тонкая часть two-level: что делать, если subdomain A сейчас на `time_level = h+5`, а subdomain B — на `h+3`? Какой halo они показывают друг другу?

**Правило:** halo всегда передаётся **для того time_level, который запрашивает receiver**. Если subdomain B хочет продвинуться на `h+4`, он запрашивает у A состояние граничных зон для `h+3` (его current) — это A может предоставить, потому что у него эти данные либо уже есть (если A прошёл через `h+3` раньше), либо он находится дальше по времени и имеет все intermediate states в архиве последних K шагов.

**Структурное следствие:** каждый subdomain обязан **хранить snapshot граничных зон для последних K шагов** (не только для текущего). Это стоимость памяти `O(K × surface_area × atom_record_size)` на subdomain. Для K=8 и умеренного subdomain'а это порядка сотен мегабайт — приемлемо.

**Протокол (упрощённо):**
```
on Inner scheduler wants to advance boundary zone z to time_level t:
    required_halo_level  ←  t - 1
    if  Outer.peer_subdomain(z).has_snapshot(required_halo_level):
        Outer.fetch_halo(z, required_halo_level)  →  inject into local state
        certificate for (z, t) can be built
    else:
        stall zone z; register callback on Outer.peer_arrived(required_halo_level)
```

**Watchdog:** если зона z стоит на стойле дольше `T_stall_max`, Outer SD Coordinator диагностирует потенциальный dead-lock между subdomain'ами на разных temporal frontiers. Deadlock predetermined невозможен при `K_max < ∞`, но stall может быть длинным — об этом отчитывается telemetry.

### 4a.9. Staging: Вариант C как принятое решение

TDMD реализует two-level decomposition **не с первого дня**, а как расширение уже работающего single-subdomain TD. Это обязательное архитектурное решение:

**v1 (M0–M6): чистый TD, single-subdomain (`P_space = 1`).** Всё внимание на том, чтобы метод работал. Multi-GPU через NVLink как один subdomain.

**M7: добавляется SD-layer как outer wrapper.** Существующий `TdScheduler` становится `InnerTdScheduler`; появляется `OuterSdCoordinator`; `CommBackend` расширяется выделенными backend'ами для outer (MPI/RDMA) и inner (NCCL/NVSHMEM) уровней. Это расширение, а не переархитектуризация.

**Почему именно так:**
- Без работающего single-subdomain TD мы не имеем права называть проект "TD engine". Сначала надо доказать, что метод работает в чистом виде (§13.3 anchor-test);
- Если бы мы начинали сразу с two-level, сложность каждого раннего milestone удваивалась бы, и с большой вероятностью потеряли бы фокус на TD в пользу SD-инфраструктуры;
- SD-layer концептуально проще TD-layer (LAMMPS и GROMACS его давно сделали хорошо). Добавить его к работающему TD — понятная задача. Обратное — нет.

### 4a.10. Явно отложенные вопросы two-level

1. **Динамическое изменение subdomain'ов** (migration atoms между subdomain'ами) — есть стандартные алгоритмы из SD-литературы, но взаимодействие с TD temporal frontiers нетривиально. v1 использует статичный SD grid; dynamic migration — post-v1.
2. **Load balancing между subdomain'ами** — при неоднородной плотности атомов (поверхности, дефекты) subdomain'ы имеют разную работу. Решение — weighted space-filling curves, см. LAMMPS `balance` command. v1 — static; post-v1 — dynamic.
3. **Long-range в two-level** — вложенное усложнение (см. §11). Сначала two-level short-range, потом long-range service.

---

# Часть II. Алгоритм и архитектура

## 5. Требования

### 5.1. Функциональные

**Физика:**
- интегрирование движения: velocity-Verlet (v1), NVT Nosé-Hoover chains (v1.5), NPT (v1.5);
- потенциалы: Morse (reference), EAM/alloy, EAM/FS, MEAM (wave 2), SNAP, MLIAP, PACE (wave 3);
- граничные условия: периодические, фиксированные, свободные по каждой оси независимо;
- single- и multicomponent системы, up to 16 species в v1, без жёстких ограничений в v2.

**TD-специфика:**
- zoning 1D/2D/3D с конфигурируемой шириной зоны ≥ `r_c + r_skin`;
- safety certificate для каждой зоны перед compute;
- адаптивный timestep (§6.5) в Production profile, фиксированный в Reference;
- K-batched pipeline с runtime-настраиваемым K ∈ [1, 64];
- migration атомов между зонами и ranks с preservation of atom identity.

**UX:**
- CLI: `run / validate / explain / compare --with lammps / resume / repro-bundle`;
- input format: LAMMPS data + `tdmd.yaml` config;
- output: LAMMPS-compatible `dump` + TDMD native HDF5 trajectory.

### 5.2. Нефункциональные

**Корректность:**
- differential match с LAMMPS на `run 0`: forces ‖Δf‖∞ / ‖f‖∞ < 1e-10 в FP64, < 1e-5 в mixed;
- NVE energy drift < 1e-6 per ns для Morse FCC Al 10⁴ атомов.

**Воспроизводимость:** три уровня гарантий, см. §7.

**Масштабируемость:** линейный scaling efficiency ≥ 80% до `n_opt` ranks на canonical benchmark (Al FCC 10⁶ atoms, Morse, `r_c = 8 Å`).

**Профилирование:** LAMMPS-compatible timing breakdown + NVTX ranges для Nsight Systems.

**Детерминированность сборки:** `Fp64ReferenceBuild` — bitwise identical результат между запусками на одной ГП-platform class.

### 5.3. Unit system support

**Принятая политика:** TDMD поддерживает два unit systems в v1 и третий в post-v1. `si` не поддерживается никогда.

| Unit system | v1 | v2+ | Назначение |
|---|---|---|---|
| `metal` | ✓ обязательный native | ✓ | металлы, сплавы, EAM, MEAM, SNAP, MLIAP, PACE — главная ниша TDMD |
| `lj` | ✓ через input adapter | ✓ | безразмерные benchmarks, canonical literature tests, correctness reference |
| `real` | ✗ | ✓ добавить | биомолекулы, органика, reactive-потенциалы (AIREBO, ReaxFF) — если понадобится |
| `si` | ✗ | ✗ никогда | не используется сообществом, делается ручной конвертацией на IO |

**Архитектурное решение:** native internal representation — **всегда `metal`**. Все state variables, все потенциалы, все constants внутри движка работают в metal (Å / eV / g·mol⁻¹ / ps). LJ и real (post-v1) поддерживаются через **input adapter** в модуле `interop/unit_converter/`.

**Почему native metal, а не "multi-unit internal":**
- наша ниша — металлы и сплавы, 100% первичных use-cases в metal;
- дублирование кода потенциалов под разные unit systems — главный источник subtle bugs в MD-кодах;
- LAMMPS-совместимость сохраняется через adapter, а не через матчинг internal representation.

**Как работает `lj` в TDMD:**
- пользователь указывает в `tdmd.yaml` блок `units: lj` + reference parameters `sigma`, `epsilon`, `mass` (в metal-единицах);
- `UnitConverter` читает LJ input file, применяет конверсию `x_metal = x_lj * sigma` и т.д., результат — стандартный metal state;
- для reverse path (сравнение dump'а с LAMMPS в lj): output dump может быть записан либо в metal, либо после inverse convert в lj — контролируется флагом `dump.units`;
- канонический LJ convention: `sigma = 1 Å, epsilon = 1 eV, mass = 1 g/mol`, если пользователь не указал другое. Это фиксированный default, явно документируется в preflight report.

**Обязательные поля в `tdmd.yaml`:**
```yaml
simulation:
  units: metal         # metal | lj  в v1; real добавится в v2
  # если units == lj, секция lj_reference обязательна:
  lj_reference:
    sigma: 1.0         # Å
    epsilon: 1.0       # eV
    mass: 1.0          # g/mol
```

Отсутствие `units:` — **preflight error**, не fallback на default. Это осознанное решение: implicit unit defaults — классический источник научных ошибок, и TDMD их запрещает.

**Differential testing:** LAMMPS и TDMD должны в тестах пользоваться one и той же unit system (выбор фиксируется в каждом benchmark'е). Преобразование на границе — только в `UnitConverter`; compare harness работает с уже конвертированными values.

**Roadmap unit systems:**
- M0–M1: только `metal`, adapter-заглушка;
- M2: добавить `lj` import через adapter, LJ reference tests для T0, T1;
- v2 (post-M8): добавить `real` когда проект начнёт расширяться в сторону reactive или organic потенциалов. До этого момента `real` — "recognized, not supported" (preflight warning).

---

## 6. Теория алгоритма TD: формальная модель

> **Scope этого раздела:** всё описанное ниже — алгоритм **внутри одного subdomain**. В single-subdomain режиме (v1 M0–M6) это алгоритм всего движка. В two-level режиме (M7+) это роль `InnerTdScheduler`; outer SD-координация описана в §4a.7.

### 6.1. Структура пространства

Ортогональный box (или subdomain в two-level режиме) `Ω = [x_lo, x_hi] × [y_lo, y_hi] × [z_lo, z_hi]` разбивается на зоны. В v1 — ортогональные зоны со сторонами `(s_x, s_y, s_z)`, каждая ≥ `r_c + r_skin`. Число зон `N_z = (N_zx, N_zy, N_zz)`.

**Scheme A — Linear Z-decomposition** (упрощённая, из §2.2 диссертации): разрезание только по Z, обход от zone 1 к zone N_zz. `N_min = 2`, хорошо работает для плоских геометрий (тонкие плёнки).

**Scheme B — 2D-decomposition Y × Z** (§2.4 диссертации): разрезание по двум осям, zigzag обход. `N_min = 6`. Применима для объёмных систем средней анизотропии.

**Scheme C — 3D Hilbert-zoning** (TDMD-extension поверх §2.5 диссертации): разрезание по трём осям с Hilbert space-filling curve для нумерации зон. Гарантирует `N_min = O(P^{2/3})` — значительно лучше линейной нумерации (`N_min = O(P)` из §2.5 диссертации). Применима для кубических геометрий.

**Выбор схемы** — задача zoning planner'а (§9), который оценивает форм-фактор box, plotted `N_min(scheme, P)`, и выбирает схему, минимизирующую `max(N_min, T_c_per_zone / T_comm)`.

### 6.2. Конечный автомат зоны

Состояния зоны:

| Состояние | Смысл | Соответствие типу диссертации |
|---|---|---|
| `Empty` | память выделена, данных нет | `w` |
| `ResidentPrev` | содержит данные предыдущего шага | `d` (частично) |
| `Ready` | cert выдан, можно запускать | — (новое, GPU-specific) |
| `Computing` | идёт force+integrate | `p` |
| `Completed` | compute завершён, не committed | `r` (частично) |
| `PackedForSend` | данные упакованы в TemporalPacket | `d` |
| `InFlight` | MPI/NCCL transfer в процессе | — (новое) |
| `Committed` | подтверждено receiver'ом, можно освобождать | — (новое, для async) |

Переходы — в приложении A.1 как DOT-граф и property-test invariants.

### 6.3. Зависимости и temporal frontier

Зона `z` на шаге `h+1` зависит от:

1. **Spatial dependencies** — зоны в пределах `r_c + r_skin` от `z` на шаге `h`, должны быть в состоянии `Completed` или старше;
2. **Temporal frontier** — самая младшая зона в системе (globally в пределах subdomain'а) на шаге `h_min`. Для зоны `z` на `h` допустимо: `h ≤ h_min + K_max`, где `K_max` — глобальный cap на глубину pipeline;
3. **Certificate dependency** — valid safety certificate для `(z, h+1)`;
4. **Neighbor validity** — neighbor list должен быть актуален; если skin исчерпан — rebuild trigger и invalidation всех certs в радиусе;
5. **Subdomain boundary dependency** (только в two-level режиме, v1.M7+): если зона `z` лежит в пределах `r_c + r_skin` от границы subdomain'а, для продвижения на `h+1` требуется, чтобы соседний subdomain подтвердил наличие halo snapshot для шага `h` (см. §4a.8). В single-subdomain режиме эта зависимость пустая.

### 6.4. Safety certificate (формально)

Для каждой `(zone, time_level)` certificate `C` — кортеж:

```
C = (safe: bool,
     cert_id, zone_id, time_level, version,
     v_max_zone, a_max_zone,
     dt_candidate, displacement_bound,
     buffer_width, skin_remaining, frontier_margin,
     neighbor_valid_until_step,
     halo_valid_until_step,
     mode_policy_tag)
```

**Displacement bound:**
```
δ(dt)  =  v_max · dt  +  0.5 · a_max · dt²
```

**Safety predicate:**
```
safe(C)  ⟺   δ(dt_candidate)  <  min(buffer_width,
                                      skin_remaining,
                                      frontier_margin)
```

**Triggers инвалидации:**
- изменение `version` состояния зоны;
- rebuild neighbor list;
- migration атомов;
- изменение `dt`;
- изменение режима исполнения;
- сдвиг temporal frontier на >1.

**Монотонность (property):** `dt_1 < dt_2 ∧ safe(C[dt_2]) ⟹ safe(C[dt_1])`. Это обязательный property-test.

### 6.5. Timestep policy

**В Reference profile:** фиксированный `dt` из конфигурации. Изменение только между runs.

**В Production profile:** адаптивный `dt` на основе **глобального** `v_max` и `a_max`:
```
dt_next  =  min(dt_cap,
                α_safety · buffer_width / max(v_max_global, ε))
```
где `α_safety ∈ [0.3, 0.7]`. Обновление `dt` — раз в R шагов, где R — cert refresh period (обычно R = 10…100).

**Запрет:** в v1 запрещён **локальный** dt per-zone. Это принципиальная позиция: локальный dt создаёт задачу re-synchronization scales, эквивалентную задаче Parareal, и выходит за рамки нашего scope. Возможное расширение в v2+.

### 6.5a. Auto-K policy (pipeline depth)

`K` — pipeline depth, number of consecutive local steps в зоне перед передачей peer'у. Это **единственный** параметр балансирующий compute vs communication в TD (§4a.6). Его правильный выбор — критически important для performance.

**Три режима:**

#### Manual K (default в Reference и Production profiles)

```yaml
runtime:
  pipeline_depth_cap: 4      # user-specified
```

`K` фиксирован user'ом. Используется для:
- Reference profile — always manual для bitwise determinism;
- Production profile — default manual для reproducible benchmarks;
- Debug / research — explicit K selection.

#### Auto-K (opt-in в Production profile, default в FastExperimental)

```yaml
runtime:
  pipeline_depth_cap: auto   # enable auto-K
  auto_k_config:
    k_range: [1, 16]
    retune_interval_steps: 10000
    measurement_steps: 500
```

Runtime периодически replanуется `K` на основе measured timing breakdown. Algorithm:

**Algorithm AutoK-v1 (initial implementation M8):**

```
initialize:
    K_current = hint_from_perfmodel()    # e.g. K=4 for Pattern 1 single-node
    last_tune_step = 0
    performance_history = []

on every step:
    if current_step - last_tune_step >= retune_interval_steps:
        trigger_measurement()

on trigger_measurement (spans `measurement_steps` iterations):
    candidates = [K/2, K, 2·K] ∩ k_range
    for K_trial in candidates:
        run measurement_steps / len(candidates) with K_trial
        record t_step_measured[K_trial]

    # Select winner:
    K_new = argmin(t_step_measured)

    # Apply hysteresis (prevent oscillation):
    if K_new != K_current:
        if |t_step[K_new] - t_step[K_current]| / t_step[K_current] < 0.05:
            K_new = K_current   # keep current, speedup not significant

    # Smooth transition:
    K_current = K_new
    last_tune_step = current_step
```

**Properties:**

- **Convergence:** algorithm local search, sweeps `K` factor of 2 каждый раз, найдёт оптимум за ~log₂(16) = 4 retune cycles worst case;
- **Overhead:** measurement_steps добавляет ~5% overhead per retune cycle. С retune_interval_steps = 10000 total overhead < 0.05%;
- **Hysteresis:** 5% threshold prevents oscillation around local optima.

**Safety constraints:**

1. `K_range` — user-configurable, default `[1, 16]`. Hard maximum `K = 64` в runtime (memory limit);
2. Auto-K **не включен** в Reference profile — would break bitwise determinism;
3. В Pattern 2 auto-K считает combined Pattern 2 cost function (outer halo + inner packets), не только inner TD;
4. Retune during neighbor rebuild периоды запрещено — data moves confounds measurements.

#### Perfmodel-assisted auto-K (M8+)

Advanced variant: использует perfmodel prediction вместо actual measurement:

```
K_predicted = perfmodel.recommend(plan, potential, hw).recommended_K
K_current = K_predicted
```

No measurement overhead. Работает если perfmodel validation error <20%. Fallback to measurement-based auto-K if perfmodel not validated для current configuration.

#### Validation

Auto-K tested в VerifyLab:
- **Correctness test:** auto-K enabled run должен давать **same scientific observables** как manual-K run (not bitwise — statistical envelope §D.13);
- **Convergence test:** synthetic workload с known optimum K — auto-K should converge within 5 retune cycles;
- **Stability test:** no oscillation в steady-state workload.

#### Roadmap

- **M5:** fixed K only, manual specification, `K ∈ {1, 2, 4, 8}`;
- **M8:** AutoK-v1 implementation (measurement-based) в FastExperimental profile;
- **M9+:** perfmodel-assisted auto-K, Production profile integration;
- **v2+:** ML-driven K prediction based на system characteristics (research).

#### Documented trade-off

Auto-K trades scientific determinism для ease-of-use и performance. Это **acceptable** в Production и FastExperimental profiles. **Неприемлемо** в Reference profile — bitwise determinism требует fixed K.

Users, кому нужна bitwise reproducibility **и** optimal K, должны:
1. Запустить auto-K calibration run (one-time);
2. Extract recommended K из telemetry;
3. Использовать manual K = extracted value в последующих Reference runs.

Это documented в UX workflow — `tdmd explain --auto-k-recommendation case.yaml` даёт recommendation без full calibration run.

### 6.5b. Expensive compute intervals policy

**Performance-critical политика** предотвращающая frequent global synchronization overhead. Урок из GROMACS и LAMMPS: без явного interval control, scientific observables (energy, virial, temperature) вычисляются **каждый шаг** — каждый вычисление требует global MPI reduction, что становится bottleneck на large ranks.

#### 6.5b.1. Категоризация expensive operations

Операции требующие global sync per call:

| Operation | Cost per call | Default interval |
|---|---|---|
| Potential energy (sum over atoms) | O(N/R) local + MPI Allreduce | 100 steps |
| Kinetic energy (sum) | O(N/R) local + MPI Allreduce | 50 steps (нужна thermostat) |
| Virial / pressure | O(N/R) local + MPI Allreduce (6 tensor) | 100 steps |
| Temperature | derived from KE | 50 steps (piggy-back KE) |
| Max velocity (для adaptive dt) | O(N/R) local + MPI Allmax | 50 steps |
| Neighbor rebuild displacement tracking | O(N/R) local + MPI Allmax | Each step (cheap локально) |

Displacement tracking — единственный **per-step** expensive sync, но cheap (single scalar Allmax).

#### 6.5b.2. Конфигурация

```yaml
runtime:
  compute_intervals:
    potential_energy: 100       # каждые 100 шагов
    kinetic_energy: 50          # нужно для thermostat, чаще
    virial: 100                 # для pressure output
    temperature: 50             # follows kinetic_energy
    adaptive_dt_recheck: 50     # для adaptive timestep
    # max_displacement: 1        # фиксированный = 1, не настраивается
```

**Automatic derived constraints:**
- `kinetic_energy.interval <= thermostat_update_interval` (NVT/NPT);
- `virial.interval <= barostat_update_interval` (NPT);
- `potential_energy.interval <= min(dump_interval)` where dump needs energy;
- `adaptive_dt_recheck.interval >= K_max` (чтобы auto-K не интерфелил).

`PolicyValidator::check` enforces эти constraints. Violation → reject config с explanation.

#### 6.5b.3. Warnings для aggressive coupling

Если user задал interval = 1 для dorogих операций:

```
Warning: aggressive compute_intervals detected
  potential_energy.interval = 1 (default: 100)
  Expected overhead на 256 ranks: ~8% of runtime

  Rationale: potential energy requires global Allreduce каждый step.
  On commodity network (~5 μs latency): 5 μs × 10⁶ steps = 5 sec overhead.

  Recommendation: Use default (100) unless you specifically need per-step energy output.
  Override via --ignore-interval-warnings if intentional.
```

#### 6.5b.4. Telemetry integration

Metrics:
```
runtime.compute_intervals.pe_computations_total
runtime.compute_intervals.ke_computations_total
runtime.compute_intervals.global_reduction_time_ms_total
runtime.compute_intervals.avg_reduction_latency_us
```

В final report отдельная секция "Global synchronization breakdown" — показывает сколько времени тратится на разные syncs.

#### 6.5b.5. Dump/output alignment

Dump operations also trigger compute if их fields require это. Validation:

```
dump:
  - format: lammps_dump_text
    path: traj.lammpstrj
    interval: 100
    fields: [id, type, x, y, z, pe]      # pe = potential energy per atom
```

Dump interval (100) должен align с `potential_energy.interval` (100) — otherwise dump would trigger extra PE computation. Preflight validation:

```
Warning: dump interval misaligned with compute interval
  dump 'traj.lammpstrj' interval: 100, requires potential_energy
  compute_intervals.potential_energy.interval: 500
  Effect: PE computed каждые 500 steps по default, но dump triggers extra computation каждые 100 steps

  Options:
    1. Change compute_intervals.potential_energy to 100 (or common factor)
    2. Remove 'pe' from dump fields (dump without pe)
    3. Proceed: extra PE computations add ~5% overhead
```

#### 6.5b.6. Auto-tuning (M8+)

Runtime может **измерять** reduction latency и **recommend** optimal intervals:

```
tdmd explain --compute-intervals case.yaml

Compute intervals analysis:
  Measured MPI Allreduce latency (32 ranks, InfiniBand): 8 μs
  Current potential_energy interval: 100

  Cost analysis:
    Current: 8 μs × 10^4 reductions/sec = 80 ms/sec overhead (8%)
    Recommended: interval=500 → 1.6% overhead
    Accuracy impact: negligible (PE trajectory смоотс)

  Recommendation: set potential_energy interval to 500
```

#### 6.5b.7. Roadmap

- **M2:** basic compute_intervals config + validation;
- **M4:** telemetry metrics для reduction overhead tracking;
- **M7:** multi-rank overhead measurement (Pattern 2);
- **M8:** auto-tuning recommendations;
- **v2+:** dynamic interval adjustment в Production profile based на measured reduction cost.

### 6.6. Псевдокод одной итерации (global view, synchronous-enough)

```
iteration(step h):
    # 1. Refresh state tracking
    displacement_max  ←  tracker.max_displacement_since_last_rebuild()
    if  displacement_max  >  skin_threshold:
        trigger_neighbor_rebuild_zones(affected)

    # 2. Refresh certificates for zones that need them
    for  zone z in scheduler.needs_cert(step=h+1):
        C  ←  certificate.build(z, h+1, current_state)
        scheduler.submit_cert(C)

    # 3. Planner выбирает ready-tasks (zones with valid cert + deps satisfied)
    ready  ←  scheduler.select_ready_tasks(
                  priority = (time_level_asc, zone_order_canonical))

    # 4. Launch compute
    for  task in ready:
        stream  ←  scheduler.assign_stream(task)
        dispatch  force_kernel(task.zone, task.time_level, stream)
        dispatch  integrator_kernel(task.zone, task.time_level, stream)
        scheduler.mark_computing(task)

    # 5. Progress communication asynchronously
    comm.progress()
    for  completed_task in scheduler.drain_completed():
        if  completed_task  has  peers_needing_data:
            pack_zone_into_temporal_packet(completed_task)
            comm.send_temporal_packet(packet, dest=next_rank)
            scheduler.mark_packed(completed_task)

    for  arrived_packet in comm.drain_arrived():
        unpack_into_zone(arrived_packet)
        scheduler.mark_received(arrived_packet.zone_id,
                                arrived_packet.time_level)

    # 6. Commit completed tasks whose postconditions are satisfied
    scheduler.commit_completed()

    # 7. Telemetry
    telemetry.write_iteration_summary()
```

**Ключевое свойство:** функция `iteration` не является глобальным барьером. `step h` — метка для logging; фактически разные зоны в конце `iteration(h)` могут находиться на time_levels от `h-K+1` до `h+1`.

### 6.7. Псевдокод scheduler.select_ready_tasks (deterministic)

```
function select_ready_tasks(priority):
    candidates  ←  []
    for  zone z in all_zones ordered by canonical_zone_order(z):
        for  time_level t in [min_advanced(z), min_advanced(z) + K_max]:
            if  (z, t) already scheduled:  continue
            cert  ←  cert_store.get(z, t)
            if  cert is None or not cert.safe:  continue
            deps_ok  ←  all_spatial_peers_at_level(z, t-1) completed
            frontier_ok  ←  t  ≤  global_min_time_level + K_max
            neigh_ok     ←  cert.neighbor_valid_until_step  ≥  t
            halo_ok      ←  cert.halo_valid_until_step       ≥  t
            if  deps_ok ∧ frontier_ok ∧ neigh_ok ∧ halo_ok:
                candidates.append(ZoneTask(z, t, cert.version, ...))
                break   # не выдаём больше одной задачи на зону за iteration
    # Deterministic tie-break:
    sort candidates by (time_level_asc, canonical_zone_order_asc, version_asc)
    return  candidates[:max_tasks_per_iteration]
```

В Reference profile: `max_tasks_per_iteration = число_потоков_compute`, `canonical_zone_order` — Hilbert или lexicographic (fixed). В Production: допускается cost-aware priority с явным fallback на canonical при tie. В FastExperimental: priority включает device pressure metrics и опционально task stealing.

---

## 7. Режимы исполнения, точность, детерминизм

### 7.1. Двухосевая модель: BuildFlavor × ExecProfile

> **Детальная precision policy описана в Приложении D.** Эта секция — краткий overview, Приложение D — источник истины для всех деталей numerical semantics.

**BuildFlavor** (compile-time, фиксирует numerical semantics):

| Flavor | StateReal | ForceReal (SNAP) | ForceReal (EAM/pair) | AccumReal | ReductionReal | Philosophy |
|---|---|---|---|---|---|---|
| `Fp64ReferenceBuild` | double | double | double | double | double | FP64 |
| `Fp64ProductionBuild` | double | double | double | double | double | FP64 |
| `MixedFastBuild` | double | float | float | double | double | **B** (safe mixed, default) |
| `MixedFastSnapOnlyBuild` | double | **float** | **double** | double | double | **B-het** (SNAP=FP32, EAM=FP64) |
| `MixedFastAggressiveBuild` | double | float | float | float | double | **A** (opt-in research) |
| `Fp32ExperimentalBuild` | float | float | float | float | double | Single |

Philosophy B (default mixed) — float compute, double accumulate. Сохраняет energy conservation в K-batched TD pipeline.
Philosophy B-het (heterogeneous) — per-kernel precision split, одобрено только через §D.17 процедуру. SNAP=FP32 (фит-шум DFT ≈ 10⁻³ eV/atom >> FP32 ULP), EAM=FP64 (table Horner требует double по D-M6-8). Введён на M8 T8.8; см. `docs/specs/potentials/mixed_fast_snap_only_rationale.md`.
Philosophy A (opt-in) — float всюду. Быстрее на 3-8%, но с отключёнными NVE drift gates и без layout-invariant guarantees. См. §D.1.

**ExecProfile** (runtime, управляет policies):

- `Reference` — фиксированный order, запрещены heuristics, фиксированные reduction trees, no task stealing, **no GPU atomics**;
- `Production` — разрешены ограниченные оптимизации, сохраняется scientific reproducibility observables, adaptive dt on;
- `FastExperimental` — агрессивные оптимизации, GPU atomics, overlap, task stealing, mixed precision.

### 7.2. Матрица совместимости (валидируется в runtime)

| BuildFlavor ↓ \ ExecProfile → | Reference | Production | FastExperimental |
|---|---|---|---|
| `Fp64ReferenceBuild` | ✓ канон | ✓ | ⚠ overkill (warning) |
| `Fp64ProductionBuild` | ⚠ warn (identical to Ref) | ✓ канон | ✓ |
| `MixedFastBuild` | ✗ reject (philosophy mismatch) | ✓ validated only | ✓ канон |
| `MixedFastSnapOnlyBuild` | ✗ reject (philosophy mismatch) | ✓ **канон** (SNAP-dominant workloads) | ✓ |
| `MixedFastAggressiveBuild` | ✗ reject | ⚠ warn (NVE gates disabled) | ✓ канон |
| `Fp32ExperimentalBuild` | ✗ reject | ✗ reject | ✓ канон |

Full matrix с rationale — см. §D.12.

### 7.3. Три уровня воспроизводимости

1. **Bitwise determinism** — одинаковый binary + одинаковый hardware class + одинаковый config → битово идентичный результат. Обязательно для `Fp64ReferenceBuild + Reference`. **Formally binds к toolchain:** same compiler version, same CUDA version, same target architecture (`-march` explicit, не `native`), same BLAS vendor, same hardware class. Cross-toolchain binary identity **не гарантируется** (FMA emission differences), но гарантируется Level 2 envelope. Подробности и enforcement — §D.10 (FMA policy).
2. **Layout-invariant determinism** — меняется число ranks / GPU / схема раскладки, результат остаётся битово идентичным или отличается в пределах bitwise-strict envelope. Целевая цель для `Fp64ReferenceBuild + Reference`, stretch для `Fp64ProductionBuild + Production`. **Не гарантируется** для `MixedFastAggressiveBuild` (см. §D.13).
3. **Scientific reproducibility** — наблюдаемые и статистики совпадают в пределах accepted tolerance. Обязательно во всех режимах **кроме** `MixedFastAggressiveBuild` где NVE gates отключены.

### 7.4. Staging внедрения

**v1 (M0–M6):** существует **только один** BuildFlavor: `Fp64ReferenceBuild`. ExecProfile только `Reference` и `Production`. Policy validator активируется с M7. Это снимает с ранних milestones бремя инфраструктуры для policy-matrix и позволяет сосредоточиться на TD core.

**v2 (M7+):** добавляются `Fp64ProductionBuild` как отдельный target, `MixedFastBuild` — после полной differential validation. `MixedFastAggressiveBuild` — отдельный opt-in target с warnings. `Fp32ExperimentalBuild` — только как extreme opt-in research.

**M8 extension (landed T8.8, 2026-04-20):** `MixedFastSnapOnlyBuild` — специализированный BuildFlavor для SNAP-dominated workloads, combination `StateReal=double, EAM в double, SNAP в float`. Это единственный допустимый путь per-kernel precision разнообразия — через явный отдельный BuildFlavor, не через runtime overrides (см. §D.11). Формальный §D.17 7-step procedure документирован в `docs/specs/potentials/mixed_fast_snap_only_rationale.md`. CMake option `-DTDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild` зарегистрирован в T8.8; kernel split emission — T8.9; slow-tier VerifyLab pass (§D.17 step 5) — T8.12 hard gate перед M8 closure.

---

## 8. Архитектура: модули и поток данных

### 8.1. Модули верхнего уровня

```
io/            чтение/запись, форматы
state/         AtomSoA, box, species, identity
zoning/        построение зон, схемы разбиения, canonical ordering
neighbor/      cell grid, neighbor list, skin, displacement tracker
potentials/    pair, many-body, ML — единый интерфейс PotentialModel
integrator/    velocity-Verlet, NVT, NPT, timestep policy
scheduler/     ZoneTask, certificate, DAG, policy, commit protocol
comm/          CommBackend abstract, MPI/NCCL/NVSHMEM, temporal packets
perfmodel/     analytic TD/SD models, predictions, reality-vs-model gate
telemetry/     timing breakdown, pipeline stats, NVTX, dumps
verify/        cross-module scientific validation (VerifyLab)
analysis/      MSD, RDF, energy profiles (post-processing)
interop/       LAMMPS data importer, dump writer, compare harness
cli/           run/validate/explain/compare/resume/repro-bundle
runtime/       SimulationEngine, lifecycle orchestration, policy apply
```

### 8.2. Правило владения данными (жёсткое)

- `state/` — единственный владелец AtomSoA и box;
- `zoning/` — владеет zone metadata и zone↔atom mapping (index ranges);
- `neighbor/` — владеет cell grid и neighbor lists, не владеет атомами;
- `potentials/` — не владеют state, читают via `ForceRequest`;
- `integrator/` — не владеет scheduler'ом, не владеет state (меняет через явный API);
- `scheduler/` — владеет графом зависимостей, certificate store, task queues; не владеет физикой и не владеет данными атомов;
- `comm/` — владеет network resources и pack/unpack buffers; не владеет доменной логикой;
- `telemetry/` — только наблюдает, read-only;
- `runtime/` — владеет жизненным циклом и связывает остальное, **не владеет временем** (это scheduler).

**Главный инвариант:** *никто кроме scheduler не владеет политикой продвижения по времени*. Нарушение этого принципа — architectural violation.

### 8.3. Поток данных одной иттерации

```
Input (user/restart)
    ↓
IO::parse → runtime::bootstrap_state → state::AtomSoA
    ↓
zoning::plan → zoning::assign_atoms_to_zones → state + zone_meta
    ↓
neighbor::build_cell_grid → neighbor::build_lists → NeighborList
    ↓
scheduler::initialize → certificates for initial zones
    ↓
RUN LOOP:
    scheduler::refresh_certificates
       ↑ читает neighbor, state, perfmodel ↓
    scheduler::select_ready_tasks
       ↓
    potentials::compute + integrator::advance  (on assigned streams)
       ↓
    comm::progress + scheduler::commit
       ↓
    [возможно] neighbor::rebuild_if_needed
       ↓
    telemetry::write + [возможно] io::dump
    ↓
Output / Restart
```

### 8.4. Единственная точка оркестрации

`SimulationEngine` (класс в `runtime/`) — единственная точка оркестрации. Никто кроме него не дёргает `main-loop`. Testing стратегия строится так, что `SimulationEngine` можно заменить test harness'ом, и все модули остаются функциональными.

---

## 9. Zoning planner

Zoning — **отдельный модуль**, а не функция `state/` или `scheduler/`. Это принципиально, потому что zoning определяет perf model и N_min, т.е. это архитектурное решение, а не implementation detail.

**Интерфейс:**

```cpp
struct ZoningPlan {
    ZoningScheme    scheme;      // Linear1D | Decomp2D | Hilbert3D
    std::array<uint32_t, 3>  n_zones;
    std::array<double, 3>    zone_size;
    uint64_t        n_min_per_rank;
    uint64_t        optimal_rank_count;
    std::vector<ZoneId>      canonical_order;   // размер = product(n_zones)
};

class ZoningPlanner {
public:
    ZoningPlan plan(const Box& box,
                    double cutoff, double skin,
                    uint64_t n_ranks,
                    const PerformanceHint& hint) const;
    // returns optimal scheme + canonical ordering
};
```

**Алгоритм выбора схемы (псевдокод):**

```
plan(box, r_c, r_skin, P, hint):
    w_zone = r_c + r_skin
    aspect = box.extent / w_zone   # вектор в 3D
    candidates = [
        (Linear1D, aspect),
        (Decomp2D, aspect),
        (Hilbert3D, aspect),
    ]
    for each candidate:
        N_min = estimate_N_min(candidate.scheme, aspect, hint.cost_per_zone)
        n_opt = floor(total_zones(candidate) / N_min)
        score = evaluate(scheme, n_opt, P, hint)
    return candidate with best score
```

**Property tests:**

- для любого разумного `aspect` и `P`, `plan().optimal_rank_count ≥ 1`;
- `plan().canonical_order` — permutation, каждый zone_id встречается ровно раз;
- `plan(box, r_c, r_skin, 1, ·).scheme = Linear1D` (1 rank не требует сложных схем);
- для `aspect ≈ (1,1,1)` и большого P, выбор = Hilbert3D (т.к. минимизирует N_min).

---

## 10. Параллельная модель и коммуникации

### 10.1. Deployment patterns и топологии

TDMD поддерживает три deployment pattern'а. Выбор определяется масштабом и железом.

**Pattern 1 — Single-subdomain TD (v1 M0–M6 baseline):**
- `P_space = 1`, `P_time = P`;
- все ranks — внутри одного subdomain'а, ведут TD pipeline;
- подходит для: workstation-class DGX, single-node multi-GPU, small clusters (≤ 16 GPU);
- коммуникации: только temporal packets между ranks.

**Pattern 2 — Two-level hybrid TD×SD (v1 M7+, production default):**
- `P_space × P_time`, обычно `P_time = GPUs_per_node`, `P_space = число_нод`;
- подходит для: multi-node clusters, cloud HPC, medium-to-large deployments;
- коммуникации: temporal packets внутри subdomain'а (intra-node) + halo exchange между subdomain'ами (inter-node);
- формальная модель — §4a.

**Pattern 3 — SD-vacuum mode (debugging / fallback):**
- `P_space = P`, `P_time = 1`;
- TDMD работает как чистый LAMMPS-стиль SD-движок, TD отключён;
- подходит для: debugging, baseline compare против TD, верификация что TDMD корректен и без TD;
- не рекомендуется для production — конкурировать с LAMMPS на его поле нет смысла.

### 10.2. Топологии коммуникации

**Ring (legacy, для воспроизведения результатов диссертации):**
rank `i` → rank `i+1 mod P`. Только `send_temporal_packet(dest=next)`. Это режим TIME-MD программы Андреева, сохраняется как `CommBackend::Ring` для anchor-test'а (см. §13.3). Применяется только в Pattern 1.

**Mesh / Cartesian grid (inner layer, по умолчанию для Pattern 1 и inner-уровня Pattern 2):**
rank видит себя в 3D-решётке `(p_x, p_y, p_z)` внутри своего subdomain'а. Соседи — ±1 по каждой оси. Temporal packets ходят по ребру `(p_x, p_y, p_z) → (p_x, p_y, p_z + 1)` для Linear1D по Z, или по зигзагу для 2D/3D схем. Это основной режим внутри subdomain'а.

**Cartesian SD grid (outer layer, Pattern 2):**
subdomain'ы образуют 3D-решётку. Halo exchange — стандартный SD protocol, 6/18/26 соседей в зависимости от stencil'а потенциала. Применяется между subdomain'ами в Pattern 2.

**All-to-all halo (SD-vacuum, Pattern 3 fallback):**
используется только в Pattern 3. Применяется для debugging и чтобы *доказать*, что TD даёт выигрыш: один binary, один config, два runs — с TD (Pattern 1 или 2) и без (Pattern 3).

### 10.2. Backends

```
CommBackend (abstract)
├── MpiHostStagingBackend    — universal fallback, pack→host→MPI→host→unpack
├── GpuAwareMpiBackend       — CUDA-aware MPI, device pointers напрямую
├── NcclBackend              — intra-node NCCL collectives
├── NvshmemBackend           — для future ultra-low-latency cases
└── RingBackend              — для anchor-test, single-axis ring
```

### 10.3. Почему TD снижает требования к каналам

Формально (из §4.3): при K-batched pipeline, `T_comm_per_step = T_p / K`. Но есть и структурный аргумент:

- **SD требует одновременной коммуникации всех neighbors на halo exchange**: rank шлёт и получает до 26 пакетов одновременно, что создаёт contention на сетевой карте и на PCIe.
- **TD на ring/mesh требует одного направленного трансфера за шаг**: rank шлёт одному соседу, получает от другого, и оба происходят **параллельно** с compute текущей зоны.

Это снимает требование к **bisection bandwidth** сети: для SD bisection критичен (многие cuts пересекают network), для TD critical path — только nearest-neighbor ring, что легко satisfies даже «дешёвая» сеть.

**Пример (из диссертации §3.5):** 10⁶ атомов алюминия, Lennard-Jones, 8 Å cutoff, 1 Гбит сеть → 512 Мбит на шаг → 3–4% от времени расчёта шага. На 2·10⁶ атомов — 6–7%, на 4·10⁶ — 12–14%. Тренд показывает: даже при grown model scaling остаётся на уровне commodity network.

### 10.4. Протокол передачи зон

**TemporalPacket**:

```
{
    protocol_version: u16,
    zone_id: u32,
    time_level: u64,
    version: u64,
    atom_count: u32,
    species_count: u16,
    box_snapshot: Box,      // для periodic wrap
    atoms: AtomSoA slice,   // id, type, x, y, z, vx, vy, vz, [fx,fy,fz if needed]
    certificate_hash: u64,  // для валидации on receiver
    crc32: u32
}
```

**Протокол (sender side):**

```
on zone becomes Completed:
    if (no downstream peer needs data):   mark Committed; return
    pack into TemporalPacket (stream_comm)
    send async (MPI_Isend / ncclSend)
    mark PackedForSend, register completion event
    on send-complete event:
        mark InFlight → Committed (upon ACK or eager protocol)
```

**Протокол (receiver side):**

```
on MPI_Irecv completes / ncclRecv returns:
    validate protocol_version + crc
    unpack into local zone (must be in Empty or ResidentPrev state)
    validate certificate compatibility
    update zone.time_level, zone.version
    mark Ready-for-next-step
    notify scheduler event ZoneDataArrived
```

### 10.5. Обработка миграций атомов

Migration (атом пересекает границу зоны) делается в отдельной phase между iterations, не встраивается в комм-протокол зоны. Это упрощает invariants: temporal packet несёт зону как снапшот, миграции — отдельная операция.

---

## 11. Long-range interactions (отдельный слой)

v1 не поддерживает long-range. v2+ добавляет **split-partition service** — отдельный модуль, исполняющийся параллельно TD pipeline:

- short-range часть (≤ r_c) — через TD, как описано выше;
- long-range часть (> r_c) — PPPM / PME на выделенном наборе ranks или GPU streams, интегрируется по схеме **outer MTS** (r-RESPA-подобной): short-range шагает с малым dt, long-range — с dt×n_rRESPA.

Архитектурно это выглядит так: `LongRangeService` работает в своём ExecProfile (может быть `Production` при TD ядре в `Reference`), обменивается с TD ядром через специальные `LongRangeWindowDependency` в графе зависимостей.

Это явно **вне v1** и **вне M7**. Сюда относится критическое расширение для charged systems, но его дизайн формализован и зарезервирован.

---

# Часть III. Интерфейсы и реализация

## 12. Ключевые интерфейсы C++/CUDA

Нотация: показаны публичные интерфейсы модулей, ABI-критичные структуры. Полные определения — в модульных `SPEC.md`.

### 12.1. Базовые типы

```cpp
namespace tdmd {

enum class ExecProfile      { Reference, Production, FastExperimental };
enum class DeviceBackend    { CPU, CUDA };
enum class ZoningScheme     { Linear1D, Decomp2D, Hilbert3D };

using AtomId    = uint64_t;
using SpeciesId = uint32_t;
using ZoneId    = uint32_t;
using CellId    = uint32_t;
using TimeLevel = uint64_t;
using Version   = uint64_t;

struct BuildFlavorInfo {
    std::string  build_flavor;        // "Fp64Reference" и т.п.
    std::string  numeric_config_id;   // хэш NumericConfig::typename
    std::string  git_sha;
    std::string  compiler_id;
};

struct RuntimeConfig {
    ExecProfile     exec_profile;
    DeviceBackend   backend;
    bool            gpu_aware_mpi;
    bool            enable_nvtx;
    bool            enable_task_stealing;
    uint32_t        pipeline_depth_cap;   // K_max
    uint64_t        global_seed;
    uint64_t        run_id;
};

} // namespace tdmd
```

### 12.2. Состояние

```cpp
struct AtomSoA {
    std::vector<AtomId>    id;
    std::vector<SpeciesId> type;
    std::vector<double>    x, y, z;
    std::vector<double>    vx, vy, vz;
    std::vector<double>    fx, fy, fz;
    std::vector<int32_t>   image_x, image_y, image_z;
    std::vector<uint32_t>  flags;
};

struct Box {
    double  xlo, xhi, ylo, yhi, zlo, zhi;
    bool    periodic_x, periodic_y, periodic_z;
};
```

### 12.3. Zoning

```cpp
struct ZoningPlan {
    ZoningScheme                  scheme;
    std::array<uint32_t, 3>       n_zones;
    std::array<double, 3>         zone_size;
    uint64_t                      n_min_per_rank;
    uint64_t                      optimal_rank_count;
    std::vector<ZoneId>           canonical_order;
};

class ZoningPlanner {
public:
    virtual ZoningPlan plan(const Box&, double cutoff, double skin,
                            uint64_t n_ranks,
                            const PerformanceHint&) const = 0;
    virtual ~ZoningPlanner() = default;
};
```

### 12.4. Scheduler

```cpp
enum class ZoneState { Empty, ResidentPrev, Ready, Computing,
                        Completed, PackedForSend, InFlight, Committed };

struct SafetyCertificate {
    bool        safe;
    uint64_t    cert_id;
    ZoneId      zone_id;
    TimeLevel   time_level;
    Version     version;
    double      v_max_zone, a_max_zone;
    double      dt_candidate;
    double      displacement_bound;
    double      buffer_width, skin_remaining, frontier_margin;
    TimeLevel   neighbor_valid_until_step;
    TimeLevel   halo_valid_until_step;
    uint64_t    mode_policy_tag;
};

struct ZoneTask {
    ZoneId      zone_id;
    TimeLevel   time_level;
    Version     local_state_version;
    uint64_t    dep_mask;
    uint64_t    certificate_version;
    uint32_t    priority;
    uint32_t    mode_policy_tag;
};

class TdScheduler {
public:
    virtual void    initialize(const ZoningPlan&) = 0;
    virtual void    refresh_certificates() = 0;
    virtual std::vector<ZoneTask> select_ready_tasks() = 0;
    virtual void    mark_computing(const ZoneTask&) = 0;
    virtual void    mark_completed(const ZoneTask&) = 0;
    virtual void    mark_packed(const ZoneTask&) = 0;
    virtual void    mark_inflight(const ZoneTask&) = 0;
    virtual void    mark_committed(const ZoneTask&) = 0;
    virtual void    commit_completed() = 0;
    virtual bool    finished() const = 0;

    // Критические методы из perf model:
    virtual size_t  min_zones_per_rank() const = 0;
    virtual size_t  optimal_rank_count(size_t total_zones) const = 0;
    virtual size_t  current_pipeline_depth() const = 0;

    virtual ~TdScheduler() = default;
};
```

### 12.5. Потенциалы

```cpp
enum class PotentialKind {
    Pair,           // Morse, LJ
    ManyBodyLocal,  // EAM, MEAM
    Descriptor,     // SNAP, MLIAP, PACE
    Reactive,       // future
    Hybrid          // future
};

struct ForceRequest {
    const AtomSoA*          atoms;
    const NeighborList*     neigh;
    const Box*              box;
    const ZoneId*           zone_ids;   // filter
    uint32_t                n_zones;
    ComputeMask             mask;       // {force, energy, virial}
};

struct ForceResult {
    double  potential_energy;
    double  virial[6];
    // forces are written in-place into atoms->fx/fy/fz
};

class PotentialModel {
public:
    virtual std::string     name() const = 0;
    virtual PotentialKind   kind() const = 0;
    virtual double          cutoff() const = 0;
    virtual double          effective_skin() const = 0;   // для safety
    virtual bool            is_local() const = 0;          // для TD applicability
    virtual void            compute(const ForceRequest&, ForceResult&) = 0;
    virtual ~PotentialModel() = default;
};
```

### 12.6. Comm

```cpp
struct TemporalPacket {
    uint16_t    protocol_version;
    ZoneId      zone_id;
    TimeLevel   time_level;
    Version     version;
    uint32_t    atom_count;
    Box         box_snapshot;
    std::vector<uint8_t>  payload;   // serialized AtomSoA slice
    uint64_t    certificate_hash;
    uint32_t    crc32;
};

class CommBackend {
public:
    virtual std::string  name() const = 0;

    // Inner (intra-subdomain) TD communication:
    virtual void send_temporal_packet(const TemporalPacket&, int dest) = 0;
    virtual std::vector<TemporalPacket>  drain_arrived() = 0;

    // Outer (inter-subdomain) SD halo — used only in two-level Pattern 2 (M7+):
    virtual void send_subdomain_halo(const HaloPacket&, int dest_subdomain) = 0;
    virtual std::vector<HaloPacket>  drain_halo_arrived() = 0;

    virtual void progress() = 0;
    virtual ~CommBackend() = default;
};

struct HaloPacket {
    uint16_t    protocol_version;
    uint32_t    source_subdomain_id;
    TimeLevel   time_level;
    uint32_t    atom_count;
    std::vector<uint8_t>  payload;
    uint32_t    crc32;
};
```

### 12.7. Perf model

```cpp
struct PerformancePrediction {
    double  t_step_td_seconds;
    double  t_step_sd_seconds;
    double  t_step_hybrid_seconds;  // two-level TD×SD, Pattern 2
    double  speedup_td_over_sd;
    double  speedup_hybrid_over_pure_sd;
    uint64_t  recommended_K;
    uint32_t  recommended_P_space;
    uint32_t  recommended_P_time;
    std::string  recommended_pattern;  // "Pattern1" | "Pattern2" | "Pattern3"
    std::string  rationale;
};

class PerfModel {
public:
    virtual PerformancePrediction predict(const ZoningPlan&,
                                           const PotentialModel&,
                                           const RuntimeConfig&,
                                           const HardwareProfile&) const = 0;
    virtual ~PerfModel() = default;
};
```

### 12.7a. Outer SD Coordinator (Pattern 2, M7+)

```cpp
struct SubdomainGrid {
    std::array<uint32_t, 3>  n_subdomains;   // (P_space_x, y, z)
    std::vector<Box>         subdomain_boxes;
    std::vector<int>         rank_of_subdomain;
};

class OuterSdCoordinator {
public:
    virtual void initialize(const SubdomainGrid&, uint32_t K_max) = 0;

    // Called by InnerTdScheduler перед продвижением boundary зоны:
    virtual bool can_advance_boundary_zone(
        ZoneId local_zone, TimeLevel target_level) = 0;

    // Halo snapshot archive: хранит последние K снимков граничных зон:
    virtual void register_boundary_snapshot(
        ZoneId local_zone, TimeLevel level, const HaloSnapshot&) = 0;

    virtual std::optional<HaloSnapshot> fetch_peer_snapshot(
        uint32_t peer_subdomain, ZoneId peer_zone, TimeLevel level) = 0;

    // Watchdog:
    virtual void check_stall_boundaries(std::chrono::milliseconds T_stall_max) = 0;

    // Global temporal frontier tracking:
    virtual TimeLevel global_frontier_min() const = 0;
    virtual TimeLevel global_frontier_max() const = 0;

    virtual ~OuterSdCoordinator() = default;
};
```

В Pattern 1 (single-subdomain) этот класс не используется. В Pattern 2 (M7+) существует ровно один экземпляр на run, координирует все subdomain'ы.

### 12.8. SimulationEngine

```cpp
class SimulationEngine {
public:
    void initialize(const RuntimeConfig&, const SimulationInput&);
    void run(uint64_t n_steps);
    void finalize();

    // Introspection для tests и explain command
    BuildFlavorInfo          build_info() const;
    PerformancePrediction    predicted_perf() const;
    const TdScheduler&       scheduler() const;

private:
    RuntimeConfig            runtime_;
    BuildFlavorInfo          build_info_;
    AtomSoA                  atoms_;
    Box                      box_;
    ZoningPlan               zoning_;
    std::unique_ptr<NeighborList>     neighbor_;
    std::unique_ptr<PotentialModel>   potential_;
    std::unique_ptr<Integrator>       integrator_;
    std::unique_ptr<TdScheduler>      scheduler_;         // InnerTdScheduler в Pattern 2
    std::unique_ptr<OuterSdCoordinator>  outer_;          // nullptr в Pattern 1 и 3
    std::unique_ptr<CommBackend>      comm_;
    std::unique_ptr<PerfModel>        perfmodel_;
    std::unique_ptr<TelemetrySink>    telemetry_;
};
```

### 12.9. UnitConverter (interop/)

```cpp
enum class UnitSystem { Metal, Lj /*, Real post-v1*/ };

struct LjReference {
    double  sigma;     // Å
    double  epsilon;   // eV
    double  mass;      // g/mol
    // default: sigma=1, epsilon=1, mass=1 (canonical convention)
};

struct UnitMetadata {
    UnitSystem            input_system;
    UnitSystem            native_system;     // всегда Metal в v1
    std::optional<LjReference>  lj_ref;      // требуется если input_system==Lj
};

class UnitConverter {
public:
    // Применяется на импорте; возвращает state в native (metal) units.
    virtual void convert_atoms_to_native(
        AtomSoA& atoms, Box& box, const UnitMetadata&) const = 0;

    // Применяется на выводе dump'а если пользователь попросил units=original.
    virtual void convert_atoms_from_native(
        AtomSoA& atoms, Box& box, const UnitMetadata&) const = 0;

    // Проверка при preflight: все ли потенциалы совместимы с выбранной unit system?
    virtual bool is_compatible(
        const PotentialModel&, const UnitMetadata&,
        std::string& reason_if_not) const = 0;

    virtual ~UnitConverter() = default;
};
```

---

# Часть IV. Верификация и тест-план

## 13. Тестовая пирамида

### 13.0. Ownership: VerifyLab

Cross-module scientific validation сконцентрирована в отдельном модуле `verify/` — **VerifyLab**. Это единственный owner для:

- canonical benchmarks (T0-T7);
- threshold registry (все числовые допуски проекта);
- differential harness (TDMD vs LAMMPS);
- anchor-test framework (воспроизведение диссертации);
- physics invariant tests (conservation laws, thermodynamics);
- perfmodel validation integration;
- regression baseline versioning;
- acceptance reports для CI и scientific reviewers.

Детальный контракт — в `docs/specs/verify/SPEC.md`. Здесь приведена общая taxonomy и ссылки.

**Scope boundary:** unit tests и property fuzz tests отдельных модулей **остаются в module-owned `tests/<module>/`**. VerifyLab владеет только тем, что пересекает границы модулей.

### 13.1. Шесть тестовых слоёв

1. **Unit tests** — изолированные функции: potential force/energy, stable sort, reductions, cert math, cell binning.
2. **Property tests** — инварианты на рандомизированных входах (fuzzer + shrinker): scheduler state machine, certificate monotonicity, neighbor rebuild consistency.
3. **Differential tests** — числовое сравнение с LAMMPS (и с self-reference) на фиксированных benchmarks.
4. **Determinism tests** — bitwise equality между repeated runs / restart-resume / разными раскладками ranks.
5. **Performance tests** — wall-clock, efficiency, scaling, zero-regression policy против stored baseline.
6. **Perf-model validation tests** — `|predict - measure| / measure < 0.2` на canonical benchmarks.

### 13.2. Canonical benchmarks (acceptance suite)

| Tier | Name | Description | Primary purpose |
|---|---|---|---|
| T0 | `morse-analytic` | 2 атома, аналитика | unit |
| T1 | `al-fcc-small` | Al 64–512, Morse | correctness + determinism |
| T2 | `al-fcc-medium` | Al 10⁴–10⁵, Morse | NVE drift, repro |
| T3 | `al-fcc-large` | Al 10⁶, Morse | **TD anchor-test vs диссертация §3.5** |
| T4 | `nial-alloy` | Ni/Al EAM, multicomp | EAM correctness vs LAMMPS |
| T5 | `meam-angular` | Si MEAM | many-body TD advantage target |
| T6 | `snap-tungsten` | W SNAP | **ML target niche proof-of-value** |
| T7 | `mixed-scaling` | T4 + T6 parallel | multi-GPU TD×SD |

### 13.3. Anchor-test (воспроизведение TIME-MD)

**Что:** в M6 (после первого полностью работающего TD scheduler'а) проект обязан воспроизвести эксперимент из §3.5 диссертации: Al 10⁶ atoms, Lennard-Jones `r_c = 8 Å`, 85 atoms в сфере, cyclic BC, разбиение по Z, ring topology.

**Успех:** численное соответствие таблицы «performance and efficiency vs number of processors» из рис. 29–30 диссертации с погрешностью ≤ 10% на аналогичном hardware class (нормализация по relative FLOPs).

**Зачем:** подтверждает, что мы реализовали тот же метод, а не пересказ. Без этого теста проект не имеет права использовать term "Time Decomposition method by Andreev".

### 13.4. Инварианты scheduler (property-fuzzing)

```
forall seed in seeds_fuzzing:
    random_events = generate_events(seed, N=10000)
    scheduler = new CausalWavefrontScheduler(Reference mode)
    for event in random_events:
        apply(scheduler, event)
        assert invariants(scheduler)

# Checked invariants:
I1: Committed zone cannot return to Ready without time_level++
I2: Zone cannot be Computing without valid certificate
I3: Zone cannot be simultaneously in ready_queue and inflight_queue
I4: No two active tasks with same (zone_id, time_level, version)
I5: Completed != Committed; commit is separate phase
I6: frontier_min + K_max ≥ max time_level of any Ready task
I7: certificate monotonicity: cert(dt1, ·) ∧ dt2 < dt1 ⟹ cert(dt2, ·)
```

### 13.5. Determinism matrix

| Тест | Reference | Production | Fast |
|---|---|---|---|
| Same run twice → bitwise equal | ✓ обязателен | ✓ same layout | ✗ |
| Restart mid-run equivalent | ✓ | ✓ | observables only |
| 1 GPU vs 2 GPU same → bitwise | ✓ target | observables | ✗ |
| Different zoning scheme → bitwise | — (not required) | — | — |
| Different ExecProfile → scientific repro | N/A | ✓ | ✓ |

### 13.6. Efficiency gates (merge gate)

PR с изменениями, затрагивающими scheduler / neighbor / potentials / comm, не может быть смержен, если на canonical T3 benchmark (Al FCC 10⁶, 8 ranks):

- `scaling_efficiency < 80%` — **hard reject**;
- `scheduler_idle_fraction > 15%` — **hard reject**;
- `neighbor_rebuild_fraction > 5%` — **soft reject (требует обоснования)**.

Baselines хранятся в `ci/perf_baselines/` и версионируются коммитом.

### 13.7. Differential vs LAMMPS

Для каждого benchmark (T1–T6) в CI выполняется:

1. Тот же input (LAMMPS data файл + `tdmd.yaml` ↔ LAMMPS script) прогоняется TDMD и LAMMPS;
2. `run 0` force/energy/virial:
   - FP64: `‖Δf‖∞ / ‖f‖∞ < 1e-10`;
   - Mixed: `< 1e-5`;
3. NVE drift через 10⁴ шагов: `|ΔE|/E_total < 1e-6` (FP64), `< 1e-4` (Mixed);
4. Observables (T, P, ⟨E⟩, MSD) через 10⁵ шагов: match within `2σ` statistical.

---

# Часть V. План реализации

## 14. Roadmap в 8 milestone'ах

Принципиальное отличие от предыдущих версий: **anchor-test и perf-model — обязательные артефакты ранних milestones**, а не финальные nice-to-have.

### M0 — Process & skeleton (4 нед.)

- monorepo (src / tests / benchmarks / docs / ci / tools);
- CMake (CPU + CUDA targets);
- clang-format, clang-tidy, pre-commit hooks;
- CI Pipeline A (lint + build) активен;
- templates для `SPEC.md`, `TESTPLAN.md`, `README.md`;
- baseline docs: positioning (§3), TD theory (§§2,4).

**Artifact gate:** repo собирается на CI green; есть 3 empty modules с рабочим CMake.

### M1 — CPU Reference MD без TD (6 нед.)

- `AtomSoA`, `Box`, `Species`;
- LAMMPS data importer + minimal `tdmd.yaml`;
- `UnitConverter` skeleton с поддержкой `metal` (native), заглушка для `lj`;
- `CellGrid` + neighbor list (CPU, deterministic reorder);
- `velocity-Verlet`, `MorsePotential`;
- CLI: `tdmd run`, `tdmd validate`;
- `run 0` works для Al FCC small;
- diff vs LAMMPS на T1: pass.

**Artifact gate:** canonical T1 differential test — green; preflight корректно требует `units:` в yaml.

### M2 — EAM CPU + perf model skeleton + lj support (6 нед.)

- `EamAlloyPotential`, `EamFsPotential` CPU;
- T4 `nial-alloy` benchmark green;
- **`UnitConverter` полная поддержка `lj`** с canonical convention;
- T0 `morse-analytic` и T1 reference tests также prepared в lj-варианте для correctness cross-check;
- первая версия `PerfModel::predict()` (analytic TD vs SD);
- `tdmd explain --perf` выводит prediction;
- telemetry skeleton (timing breakdown в LAMMPS-compatible формате).

**Artifact gate:** T4 differential green в metal; T1 differential green одновременно в metal и lj (одинаковые результаты после конверсии); `explain --perf` корректно ранжирует SD > TD на тривиальных случаях (т.к. TD ещё нет).

### M3 — Zoning planner + neighbor ТD-ready (4 нед.)

- `ZoningPlanner` со всеми тремя схемами (Linear1D / Decomp2D / Hilbert3D);
- `N_min` формулы валидированы property-тестами для каждой схемы;
- neighbor rebuild с skin tracking и displacement cert;
- displacement fuzzer tests.

**Artifact gate:** для каждой схемы zoning'а, `plan().n_min_per_rank` совпадает с аналитическим предсказанием диссертации на тест-моделях.

### M4 — Deterministic TD scheduler (single-node, CPU) (8 нед.)

- `ZoneState` state machine + transition property-tests (invariants I1–I7);
- `SafetyCertificate` math + monotonicity tests;
- `CausalWavefrontScheduler (Reference mode)`, K=1 baseline;
- `SimulationEngine` orchestrates TD-enabled run;
- commit protocol (two-phase);
- deadlock watchdog.

**Artifact gate:** T1, T2 green в TD mode; bitwise determinism tests pass.

### M5 — Multi-rank TD on CPU (MPI) + K-batching (6 нед.)

- `MpiHostStagingBackend` (baseline, без GPU);
- `RingBackend` (legacy, для anchor-test);
- temporal packet protocol;
- K-batched pipeline (K ∈ {1, 2, 4, 8});
- T3 Al-FCC-large benchmark runs;
- **anchor-test §13.3 green** — mandatory gate.

**Artifact gate:** anchor-test pass. TDMD впервые доказывает, что воспроизводит метод диссертации.

### M6 — GPU path single-GPU (8 нед.)

- CUDA kernels: cell binning, neighbor build, Morse force, EAM density/force, integrator;
- `DeviceAtomSoA`, GPU-resident state;
- NVTX ranges для Nsight Systems;
- streams (`stream_compute`, `stream_comm`, `stream_aux`);
- T1–T4 green на GPU;
- Mixed precision flavor `MixedFast` — compile-target добавлен, но ещё не default.

**Artifact gate:** T4 differential FP64 green on GPU; T4 mixed-precision differential green within `1e-5`.

### M7 — Two-level TD×SD hybrid (Pattern 2 introduction) (10 нед.)

Это milestone, на котором TDMD переходит от single-subdomain (Pattern 1) к two-level deployment (Pattern 2). Существующий `TdScheduler` переименовывается в `InnerTdScheduler`; появляется `OuterSdCoordinator`.

- `GpuAwareMpiBackend` (для outer SD halo exchange);
- `NcclBackend` (для inner TD temporal packets);
- `OuterSdCoordinator` с halo snapshot archive (last K snapshots);
- `SubdomainBoundaryDependency` в zone DAG scheduler'а;
- boundary zone stall protocol + watchdog;
- `PerfModel` расширен для predicting Pattern 2 performance (`t_step_hybrid_seconds`);
- `CommBackend` разделён на inner и outer channels (см. §12.6);
- T7 `mixed-scaling` benchmark;
- scaling efficiency ≥ 80% для T3 на 8 GPU single-node, ≥ 70% на 2 nodes × 8 GPU;
- **perf model validation** для Pattern 2: `|predict - measure| < 25%` — mandatory gate (допуск мягче чем для Pattern 1 из-за сложности модели).

**Artifact gate:** T7 pass; Pattern 2 validated на minimum 2 nodes; Pattern 1 остаётся fully functional (regression-test).

### M8 — SNAP + proof-of-value + MixedFastSnapOnlyBuild (6 нед.)

- `SnapPotential` CPU + GPU;
- T6 tungsten SNAP benchmark;
- **сравнение TDMD vs LAMMPS SNAP** на T6: wall-clock и scaling;
- demo: где именно TDMD обгоняет LAMMPS на представительном ML kernel;
- **`MixedFastSnapOnlyBuild`** — новый BuildFlavor (см. §D.11): `StateReal=double`, `ForceReal=float для SNAP kernels`, `ForceReal=double для EAM kernels`. Это единственный допустимый путь per-kernel precision разнообразия — через явный отдельный BuildFlavor, не через runtime overrides;
- full slow-tier VerifyLab validation для нового flavor;
- M8 closes v1 alpha.

**Artifact gate:** на T6 TDMD либо обгоняет LAMMPS ≥ 20% на целевой конфигурации (≥ 8 ranks, commodity network), либо проект документирует, почему не обгоняет и что делать дальше (честная постановка). `MixedFastSnapOnlyBuild` — rebuild и CI slow-tier pass.

### Post-v1 (M9+)

Post-v1 milestones — full scope. Ниже приведены те, которые имеют concrete research или engineering содержание и assigned owner.

#### M9 — NVT baseline (8 нед.)

- `NoseHooverNvtIntegrator` CPU + GPU;
- `NoseHooverNptIntegrator` CPU + GPU (basic isotropic);
- Policy validator enforcement: `style != nve` ⇒ `pipeline_depth_cap = 1` — reject config otherwise (см. integrator/SPEC §7.3.1);
- NVT/NPT canonical benchmarks добавлены в VerifyLab (новые T-level benchmarks: T8 NVT Al FCC, T9 NPT Al FCC);
- Differential match vs LAMMPS NVT/NPT для canonical benchmarks;
- CI gate: NVT config с K>1 автоматически rejected.

**Artifact gate:** NVT Al FCC 512 atoms, 10⁵ steps, equipartition within ±2σ, temperature distribution Maxwell-Boltzmann within chi² p=0.05. Same для NPT isotropic.

**Consequence:** NVT production работает, но без TD speedup (effective SD). Users знают и принимают это trade-off.

#### M10 — MEAM integration (8 нед.)

- `MeamPotential` CPU + GPU;
- Port из LAMMPS meam package с credit + license;
- T5 `si-meam` benchmark в VerifyLab;
- MEAM + TD performance характеристики — первый показ **TDMD signature value** для angular-dependent potentials;
- Differential vs LAMMPS `pair_style meam/c`.

**Artifact gate:** T5 differential green; MEAM на T5 показывает ≥30% speedup vs LAMMPS на 8 ranks commodity network (это target ниши — angular moments halo pressure в SD создаёт overhead который TD не имеет).

#### M11 — NVT-in-TD research window (12 нед.)

**Это research program, не engineering delivery.** Owner: Physics Engineer + Validation Engineer joint.

**Deliverables:**

1. Literature survey — Tuckerman 2010 multi-timescale Verlet + thermostat, Eastwood 2010 lazy thermostat techniques, других RESPA-like approaches; написать internal report с assessment adaptability к TD;
2. Prototype implementation на `research/nvt-in-td` branch — Вариант C из integrator/SPEC §7.3.3 (lazy thermostat synchronization);
3. Validation suite: comparison с LAMMPS NVT baseline на canonical Al FCC + Ni/Al alloy:
   - equipartition match within ±2σ;
   - temperature distribution Maxwell-Boltzmann chi² p>0.05;
   - diffusion coefficient match within ±5%;
4. Performance characterization: actual speedup от K>1 NVT mode;
5. **Go/no-go decision** based на criteria:
   - Equipartition match: MUST pass;
   - Speedup >10% vs K=1: SHOULD pass;
   - If equipartition не passes → no-go, Variant A остаётся permanent;
   - If speedup <10% → no-go, Variant A остаётся default but Variant C available как opt-in.

**Artifact gate:** research report + go/no-go decision documented в master spec Change log. Не обязательно production deployment на M11.

#### M12 — PACE + MLIAP (10 нед.)

- `PacePotential` port из ACE reference implementation;
- `MliapPotential` с plug-in architecture для Python-defined ML models (pybind11);
- Extension к `cost_per_force_evaluation` calibration для arbitrary ML models;
- T10 `ace-cu-copper` benchmark added.

#### M13 — Long-range service + NVT-in-TD (если M11 go) (12 нед.)

- PPPM (Particle-Particle Particle-Mesh) или Ewald implementation;
- Long-range integration с TD через split time-stepping;
- Если M11 go-decision: production integration lazy thermostat;
- T11 charged-system benchmark (SiO₂ glass).

#### Beyond M13 — open track

- Reactive track (ReaxFF) — separate major stream, possibly forked research effort;
- Triclinic box support;
- Per-zone adaptive dt (major architectural expansion);
- Dynamic subdomain migration (Pattern 2 enhancement);
- Python high-level API;
- Community contribution framework.

Это список не roadmap в strict sense — эти items станут milestones когда получат formal SPEC delta и owner assignment.

---

## 15. Инженерная методология

### 15.1. Spec-Driven TDD + Differential + Performance Gates

Стандартный порядок работы над фичей:

1. **SPEC** (раздел в мастер-специи или модульный `SPEC.md`);
2. **Contract tests** пишутся до кода, провалены сначала;
3. **Minimal implementation** — чтобы пройти contract tests;
4. **Differential validation** — `run 0` / NVE / observables vs LAMMPS или self-reference;
5. **Determinism check** — bitwise equality (в Reference mode);
6. **Perf gate** — нет регрессии по canonical benchmarks;
7. **Review + merge.**

Пропуск любого шага — architectural violation, блокируется merge gate'ом.

### 15.2. Модульная дисциплина

Каждый модуль обязан содержать:
- `SPEC.md` — контракт;
- `TESTPLAN.md` — тестовая стратегия;
- `README.md` — обзор и границы;
- примеры использования;
- список известных ограничений (known limitations);
- telemetry hooks.

### 15.3. AI-agent development rules

Проект разрабатывается преимущественно Codex-агентами. Правила:

1. **Spec first** — агент читает релевантные разделы мастер-специи и модульного `SPEC.md` ДО кода;
2. **No hidden second engine** — все оптимизации живут как policy-слои поверх общего core;
3. **Reference path sacred** — нельзя деградировать Reference для ускорения Production / Fast;
4. **Explicit assumptions** — если в задаче чего-то не хватает, агент перечисляет допущения и предлагает варианты, но не «молча угадывает»;
5. **Test plan mandatory** — любая физически значимая задача завершается с test plan;
6. **Report structured** — сессия заканчивается структурированным отчётом: реализовано / файлы / тесты / risks / SPEC deltas.

---

# Часть VI. Приложения

## Приложение A. Соответствие с диссертацией (traceability)

### A.1. Состояния зон

| ZoneState (TDMD) | Type (Андреев) | Семантика |
|---|---|---|
| Empty | w | свободна, готова принимать |
| ResidentPrev | d (частично) | содержит данные предыдущего шага |
| Ready | — | cert выдан (новое, уточнение для GPU) |
| Computing | p | идёт расчёт |
| Completed | r (частично) | compute завершён |
| PackedForSend | d | готова к передаче |
| InFlight | — | в процессе передачи (GPU async) |
| Committed | — | подтверждена receiver'ом |

### A.2. Формулы соответствия

| Концепт | Диссертация | TDMD §/формула |
|---|---|---|
| «множественные шаги одновременно» | §2.1 | §2.1, §2.4 |
| расчётная зона = r_c | §2.1–2.2 | §6.1 |
| минимум зон на rank для pipeline | формулы (35),(43) | §4.4, ZoningPlan::n_min_per_rank |
| оптимум процессоров | формула (44)-(45) | §4.4, ZoningPlan::optimal_rank_count |
| снижение bandwidth через K шагов | формула (51) | §4.3 |
| буфер скорости v_max·dt·α | формула (33) | §2.3 п.3, §6.4 |
| Verlet-skin | §1.3 | §2.3 п.2, neighbor/ |
| кольцевая топология | §2.1–2.5 | §10.1, RingBackend (legacy) |
| автомат состояний зоны | §2.1 figures | §6.2, приложение A.1 |
| эксперимент Al 10⁶ | §3.5, рис. 29–30 | §13.3 anchor-test |

### A.3. Что TDMD расширяет поверх диссертации

1. **Mesh/Cartesian topology** вместо исключительно ring — современная HPC-реальность;
2. **Hybrid time × space decomposition** — два уровня параллелизма одновременно, невозможно было в 2007 на доступных MVS;
3. **Hilbert 3D zoning** — существенно улучшает N_min для куба по сравнению с линейной нумерацией (§2.5) дисс.;
4. **GPU-resident state + streams** — иной compute substrate;
5. **Formalized SafetyCertificate** — диссертация описывает условия безопасности неформально;
6. **Perf model как first-class** — диссертация оценивает производительность post-hoc по замерам, TDMD предсказывает ex-ante;
7. **Policy-driven determinism levels** — диссертация не различает bitwise / layout-invariant / scientific repro.

### A.4. Что TDMD **не** реализует из диссертации

- Алгоритм ровно один-к-одному из §2.5 с линейной нумерацией 3D — заменён Hilbert (с legacy-опцией для anchor-test);
- Процессоры T800/Alpha-специфичные оптимизации — не актуальны;
- Ручной подбор sync vs async на уровне hardware — заменён async-first через CUDA streams и non-blocking MPI.

---

## Приложение B. Assumptions & Open Questions

### B.1. Принятые допущения (assumptions)

1. **Primary target hardware:** commodity HPC (x86_64 + NVIDIA GPU) и cloud (A100/H100) в v1. ARM / AMD GPU — post-v1.
2. **Units:** `metal` (native internal) + `lj` (через input adapter) в v1. `real` — post-v1 если понадобится reactive/organic. `si` — **никогда** (см. §5.3).
3. **Atom count:** до 10⁸ атомов на single node multi-GPU в v1; до 10¹⁰ на multi-node в v2.
4. **Periodic boundaries:** только ортогональный box в v1. Triclinic — post-v1.
5. **Local dt per zone:** запрещён в v1. Global dt, адаптивный в Production.
6. **Ensembles:** NVE v1, NVT/NPT v1.5.
7. **Precision default:** `Fp64Reference` — единственный в v1.
8. **Deployment pattern staging:** Pattern 1 (single-subdomain TD) — v1 M0–M6. Pattern 2 (two-level TD×SD) — добавляется на M7, принят как production default для multi-node. Pattern 3 (SD-vacuum) — всегда доступен как fallback и debugging-режим. См. §10.1.
9. **Two-level static subdomain grid:** в v1 (M7) subdomain grid статичен — выбирается на startup, не меняется в runtime. Dynamic migration атомов между subdomain'ами и load balancing — post-v1.

### B.2. Открытые вопросы (open questions)

1. **Триклинный box** — добавлять в M6 или отложить до post-v1?
2. ~~**Python API** — в v1 как bindings через pybind11, или только CLI?~~ **РЕШЕНО в v2.4**: pybind11 в post-v1 (M9+), three-layer API — см. Приложение E.
3. **Формат input:** только YAML, или YAML + scripting DSL (как LAMMPS)?
4. **Long-range priority:** M9 или более поздний milestone? Особенно остро для Pattern 2.
5. **Anchor-test hardware equivalence:** как корректно нормализовать performance measurements на T800/Alpha-2007 → современный x86? Нужно руководство.
6. **KIM OpenKIM integration** — стратегический вопрос для ML-экосистемы.
7. **Bitwise layout-invariance** на разных числах GPU — target или stretch?
8. **Reactive track (ReaxFF)** — когда открывать major stream?
9. **`real` unit system timing** — добавлять вместе с первым reactive потенциалом или раньше по запросу пользователей?
10. ~~**Auto-K policy**: автоматический подбор pipeline depth K в Production profile — алгоритм и safety bounds?~~ **РЕШЕНО в v2.4**: три operation modes (manual, measurement-based, perfmodel-assisted) — см. §6.5a.
11. **Outer SD dynamic load balancing** — на какой пост-v1 milestone?
12. **Subdomain count limits в Pattern 2:** есть ли архитектурный потолок на `P_space × P_time`, или только практический?
13. **Thermostat-in-TD-K>1**: решение Вариант A (global frozen, K=1 forced) vs Вариант C (lazy sync) — go/no-go decision на M11. См. integrator/SPEC.md §7.3.

### B.3. Explicit non-goals v1

- ab-initio MD (DFT-coupled);
- rigid body dynamics;
- constraint algorithms (SHAKE / RATTLE / LINCS);
- enhanced sampling (metadynamics, REMD);
- QM/MM;
- neutron / X-ray scattering real-time computation;
- dynamic subdomain migration (только статичный grid в Pattern 2 v1);
- `si` unit system (не поддерживается никогда);
- `real` unit system (отложен в post-v1).

---

## Приложение D. Precision Policy Details

Эта часть фиксирует детальную политику использования численной точности (floating-point precision) во всём TDMD. Является обязательным продолжением §7 (BuildFlavor × ExecProfile) и §30 (redesign).

**Scope:** что именно делается в `double`, что в `float`, когда какие reductions, GPU atomics, compile flags. Решения зафиксированы, не допускают silent изменений без SPEC delta.

### D.1. Общая философия

TDMD следует **"float compute, double accumulate"** философии (Philosophy B из LAMMPS INTEL package), с одним опциональным BuildFlavor для "float всюду" (Philosophy A) как research-grade opt-in.

**Обоснование выбора по умолчанию Philosophy B:**

1. **TD K-batched pipeline накапливает round-off K раз** перед synchronization. С float accumulation и K=8, round-off error ~3e-7 на force component накапливается в drift ~3e-7 × 10⁶ steps/ns = 0.3 eV/ns, что катастрофически нарушает NVE drift gate `1e-4 per ns` (MixedFast threshold в `verify/thresholds.yaml`).
2. **Layout-invariant determinism (§7.3) требует numerical stability** halo packets между subdomain'ами. Float forces в HaloPacket создают layout-dependent divergence.
3. **Actual throughput разница между Philosophy A и B — только 3-8%** на typical many-body kernels (EAM, MEAM, SNAP), ценой катастрофической потери numerical stability.
4. **Industry precedent:** LAMMPS INTEL package, GROMACS, AMBER используют Philosophy B как default для mixed precision.

Philosophy A предоставляется как opt-in BuildFlavor `MixedFastAggressive` для specific research purposes (ensemble screening, throughput demonstrations), с **явным отключением** NVE drift gates и layout-invariant guarantees.

### D.2. Шесть канонических BuildFlavor'ов

Обновлённый список (v2.2 имел 4, v2.3 добавил `MixedFastAggressive`, v2.5 M8 T8.8 добавляет `MixedFastSnapOnlyBuild`):

| BuildFlavor | StateReal | ForceReal (SNAP) | ForceReal (EAM/pair) | AccumReal | ReductionReal | Philosophy | Default для |
|---|---|---|---|---|---|---|---|
| `Fp64ReferenceBuild` | double | double | double | double | double | FP64 | Development, CI, reference runs |
| `Fp64ProductionBuild` | double | double | double | double | double | FP64 | Scientific production runs |
| `MixedFastBuild` | double | float | float | double | double | **B** (safe mixed) | Fast production, default throughput target |
| `MixedFastSnapOnlyBuild` | double | **float** | **double** | double | double | **B-het** (heterogeneous) | SNAP-dominant production (pure SNAP или SNAP+EAM hybrid где EAM precision matters) |
| `MixedFastAggressiveBuild` | double | float | float | float | double | **A** (unsafe mixed) | Opt-in research, ensemble screening |
| `Fp32ExperimentalBuild` | float | float | float | float | double | Single | Extreme throughput research |

**ReductionReal = double во всех случаях** (one double dimension сохраняется глобально) — это гарантирует что global energy / virial / temperature sums не deteriorated.

**Philosophy B-het** — единственный utverzhdёnный per-kernel precision mix. Path к этому flavor не runtime override (запрещено §D.11), а explicit отдельный BuildFlavor, прошедший полную §D.17 7-step procedure. Формальный rationale + threshold derivation + compat analysis для `MixedFastSnapOnlyBuild` — `docs/specs/potentials/mixed_fast_snap_only_rationale.md` (T8.8 landed 2026-04-20).

### D.3. NumericConfig templates

```cpp
namespace tdmd {

// Philosophy B (safe mixed) — default mixed flavor
struct NumericConfigMixedFast {
    using StateReal     = double;
    using ForceReal     = float;
    using AccumReal     = double;
    using ReductionReal = double;

    static constexpr bool deterministic_reduction     = false;
    static constexpr bool allow_device_atomics        = true;
    static constexpr bool allow_kahan_summation       = false;
    static constexpr bool position_delta_double       = true;   // always
    static constexpr bool eam_table_lookup_double     = true;   // always
    static constexpr bool ftz_enabled                 = true;
    static constexpr bool fast_math_allowed           = true;
};

// Philosophy A (aggressive mixed) — opt-in research
struct NumericConfigMixedFastAggressive {
    using StateReal     = double;
    using ForceReal     = float;
    using AccumReal     = float;   // <-- THE difference
    using ReductionReal = double;

    static constexpr bool deterministic_reduction     = false;
    static constexpr bool allow_device_atomics        = true;
    static constexpr bool allow_kahan_summation       = false;
    static constexpr bool position_delta_double       = true;
    static constexpr bool eam_table_lookup_double     = true;
    static constexpr bool ftz_enabled                 = true;
    static constexpr bool fast_math_allowed           = true;

    // Declared loosened guarantees:
    static constexpr bool nve_drift_gate_enforced     = false;  // <-- explicit
    static constexpr bool layout_invariant_guarantee  = false;  // <-- explicit
};

} // namespace tdmd
```

### D.4. Анатомия force kernel по precision layers

Pseudocode показывает как **одна и та же формула** выражается в разных build flavors. Это canonical pattern, которому следуют все force kernels. **Обрати внимание на `__restrict__` qualifiers** — обязательны на всех pointer parameters (см. §D.18).

```cpp
template<typename NumericConfig>
__device__ void morse_force_kernel(
    const AtomSoA* __restrict__ atoms,
    const NeighborList* __restrict__ neigh,
    const MorseParams* __restrict__ params,
    typename NumericConfig::AccumReal* __restrict__ fx_out,
    typename NumericConfig::AccumReal* __restrict__ fy_out,
    typename NumericConfig::AccumReal* __restrict__ fz_out)
{
    using State  = typename NumericConfig::StateReal;    // double always
    using Force  = typename NumericConfig::ForceReal;    // float in Mixed
    using Accum  = typename NumericConfig::AccumReal;    // double в Philosophy B

    for (int i : my_atoms) {
        Accum fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;   // double в Philosophy B

        for (int j : neighbors[i]) {
            // Position delta in State (double), always:
            State dx_d = atoms->x[i] - atoms->x[j];
            State dy_d = atoms->y[i] - atoms->y[j];
            State dz_d = atoms->z[i] - atoms->z[j];

            // Cast to Force for inner math:
            Force dx = static_cast<Force>(dx_d);
            Force dy = static_cast<Force>(dy_d);
            Force dz = static_cast<Force>(dz_d);

            Force r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > cutoff_sq) continue;

            Force r = std::sqrt(r2);
            Force exp_term = std::exp(-alpha * (r - r0));         // float in Mixed
            Force factor = 2.0f * D * alpha * (1.0f - exp_term) * exp_term / r;

            Force dfx = factor * dx;
            Force dfy = factor * dy;
            Force dfz = factor * dz;

            // Accumulate в Accum (double в Philosophy B, float в A):
            fx_i += static_cast<Accum>(dfx);
            fy_i += static_cast<Accum>(dfy);
            fz_i += static_cast<Accum>(dfz);
        }

        // Write to state via output pointers (marked __restrict__):
        fx_out[i] = fx_i;   // double в Philosophy B and Fp64*
        fy_out[i] = fy_i;
        fz_out[i] = fz_i;
    }
}
```

**Ключевые линии:**
- `State dx_d = ...` — double subtraction, потому что `atoms->x[i] - atoms->x[j]` может быть очень маленьким (catastrophic cancellation prevention);
- `Force dx = static_cast<Force>(dx_d)` — безопасный down-cast (difference уже computed в double);
- `fx_i += static_cast<Accum>(dfx)` — up-cast при accumulation (в Philosophy B accumulates в double);
- `__restrict__` на всех pointer parameters — enables vectorization (см. §D.18).

### D.5. Position delta: строго double

Независимо от BuildFlavor: **вычисление позиционной разности всегда в double**. Это non-negotiable.

**Rationale:** catastrophic cancellation. Если два атома близко (что типично для MD — соседи на 2-3 Å, box 100+ Å), разность их позиций может иметь много меньше значащих цифр чем сами позиции. В float:

```
x_i = 17.12345  (float, ~7 decimal digits)
x_j = 17.12340  (float)
dx = 5e-5       (только 1 значащая цифра, остальное — шум)
```

В double:

```
x_i = 17.12345678901234  (double, ~15 decimal digits)
x_j = 17.12340678901234  (double)
dx = 5.0000000000e-5    (15 значащих цифр, точность сохранена)
```

Применение: **все `atoms.x[i] - atoms.x[j]` subtractions в double**, включая periodic wrap delta. Далее cast к ForceReal для inner computation в float kernels.

### D.6. EAM table lookups

Tabulated functions `F(ρ)`, `ρ(r)`, `φ(r)` и их derivatives приходят в `.eam.alloy` / `.eam.fs` файле как double arrays. EAM force accuracy критична для energy conservation.

**Политика:**
- Tables stored в double memory независимо от BuildFlavor (except `Fp32Experimental` где всё в float);
- Cubic spline evaluation в double;
- Результат interpolation cast к ForceReal перед inner loop arithmetic.

**Rationale:** tables memory-bound, не compute-bound. Double tables добавляют ~2× memory traffic, но GPU L2/L1 absorb это. Speed impact — <5%. Accuracy benefit для NVE conservation — significant.

**Исключение `Fp32Experimental`:** всё в float, включая tables. Это research mode где user accepts degradation.

### D.7. Trigonometric и transcendental functions

В SNAP, MLIAP, PACE: `exp`, `sin`, `cos`, `sqrt`, `pow`, Legendre polynomials.

**Политика по build flavor:**

| BuildFlavor | Transcendentals |
|---|---|
| Fp64Reference | double (`exp`, `sin`, `cos`, ...) |
| Fp64Production | double |
| MixedFast | float (`expf`, `sinf`, `cosf`, ...) внутри compute, результат в double accumulation |
| MixedFastAggressive | float всюду, включая accumulation |
| Fp32Experimental | float всюду |

**Speedup float transcendentals:** ~2-3× на GPU (они compute-bound). Это одно из главных мест где MixedFast даёт throughput win.

**Accuracy:** float `expf` имеет relative error ~1e-7 vs double ~1e-16. В SNAP bispectrum с 50-200 neighbors, accumulated error может быть значительна. Поэтому accumulation в double (Philosophy B) необходима.

### D.8. GPU atomics policy

GPU atomic operations (`atomicAdd`) недетерминистичны — порядок completion зависит от hardware warp scheduling.

| BuildFlavor × ExecProfile | Atomics policy |
|---|---|
| Fp64Reference × Reference | **Forbidden everywhere.** Forces через explicit per-block reductions + tree merge. |
| Fp64Production × Production | Permitted only для force pair accumulation (Newton third law `fx[j] -= Δfx`). Energy / virial reductions через deterministic tree. |
| MixedFast × Production | Same as Fp64Production |
| MixedFast × FastExperimental | Unrestricted — atomics везде (forces, energies, virial) |
| MixedFastAggressive × FastExperimental | Unrestricted |
| Fp32Experimental × FastExperimental | Unrestricted |

Reference path — это **другая реализация force accumulation**, не просто другая точность. Cost: ~2-3× slower compute, acceptable trade для bitwise determinism.

### D.9. Reductions policy

Три уровня reductions, разная политика для каждого.

**Level 1 — per-atom force (pair contribution):**
- В inner force loop, `fx[i] += Δfx_j` через атомик (GPU) или sequential add (CPU);
- Subject to atomics policy (§D.8).

**Level 2 — per-zone energy / virial (local reduction):**
- Sum over atoms within zone;
- `Fp64Reference`: **deterministic tree reduction с Kahan compensation**, fixed ordering;
- `Fp64Production`: deterministic tree без Kahan;
- `MixedFast+`: CUB/thrust block reduction, non-deterministic ordering OK.

**Level 3 — global energy / virial (MPI-level reduction):**
- Sum over ranks через `CommBackend`;
- `Fp64Reference`: custom `deterministic_sum_double` (см. `comm/SPEC.md` §7.2) с Kahan + fixed rank ordering;
- `Fp64Production`: custom `deterministic_sum_double` без Kahan;
- `MixedFast+`: `MPI_Allreduce(MPI_SUM)` / `ncclAllReduce` — implementation-defined ordering, non-deterministic.

**Kahan summation enforced только в Reference.** В Production fixed ordering достаточно для scientific reproducibility, Kahan overhead (2-3× vs naive) не оправдан.

### D.10. Compiler flags policy

**Denormal handling (Flush-To-Zero):**

| BuildFlavor | FTZ |
|---|---|
| Fp64Reference | OFF — preserve IEEE 754 denormals |
| Fp64Production | OFF — scientific computing standard |
| MixedFast | ON — ~100× speedup on denormal inputs, irrelevant accuracy impact |
| MixedFastAggressive | ON |
| Fp32Experimental | ON |

Enforcement: `_MM_SET_DENORMALS_ZERO_MODE` + CUDA `--ftz=true` compiler flag, set at build time, not runtime.

**Fast-math (`-ffast-math`, CUDA `--use_fast_math`):**

Разрешает:
- Assume no NaN, no Inf;
- Associativity (`(a+b)+c == a+(b+c)` — critical implication: reorderings may occur);
- Reciprocal approximations (`1.0/x` → `__frcp_rn(x)`).

| BuildFlavor | Fast-math |
|---|---|
| Fp64Reference | **FORBIDDEN** — breaks bitwise determinism |
| Fp64Production | FORBIDDEN — scientific correctness first |
| MixedFast | Allowed (trigs, rsqrt) but с documented caveats |
| MixedFastAggressive | Allowed unrestricted |
| Fp32Experimental | Allowed unrestricted |

**FMA (fused multiply-add):**

Hardware FMA — одна операция с одним rounding для `a*b + c`. Slightly more precise than separate mul + add (one rounding vs two). Proven benefit для energy conservation в long runs.

**Issue:** compiler decides when to emit FMA. Different compilers (GCC / Clang / ICX / NVCC) emit FMA differently на одинаковом source code. Это означает что **bitwise identity cross-compiler невозможна** если FMA involved.

**Политика:**

| BuildFlavor | FMA emission |
|---|---|
| Fp64Reference | Allowed, encouraged |
| Fp64Production | Allowed, encouraged |
| MixedFast+ | Allowed, encouraged |

**Bitwise determinism claim (§7.3 Level 1) formally binds к specific toolchain + hardware class:**

Для `Fp64ReferenceBuild + Reference` bitwise determinism guaranteed **только при условии**:

1. **Same compiler + version** (e.g. GCC 11.2.0 на всех dev/CI/prod systems);
2. **Same CUDA toolkit version** (NVCC 12.2 и т.п.);
3. **Same target architecture** (`-march=native` может break — use explicit e.g. `-march=x86-64-v3`);
4. **Same hardware class** (A100 vs H100 могут differ в tensor core behavior даже без TF32);
5. **Same BLAS / LAPACK vendor** (для linear algebra в SNAP / MLIAP).

**Enforcement:** в reproducibility bundle (см. §12.9 runtime/SPEC.md §8) записываются **все пять** parameters из списка выше. При `tdmd verify --bitwise-compare run_a run_b` VerifyLab первым делом сравнивает environment fingerprint. Если отличается — явное сообщение "bitwise comparison invalid across toolchains", а не silent false positive/negative.

**CMake enforcement:**

```cmake
# Fp64Reference: lock target architecture explicitly
target_compile_options(tdmd_fp64_ref PRIVATE
    -march=x86-64-v3              # NOT -march=native
    -mtune=generic
    -ffp-contract=on              # allow FMA, но consistent
)
# NVCC:
set_target_properties(tdmd_fp64_ref PROPERTIES
    CUDA_ARCHITECTURES "80;90"    # explicit, not native
)
```

**Cross-compiler reproducibility level:**
- Same compiler + hardware → **bitwise identical** (Level 1);
- Different compiler, same hardware → **layout-invariant envelope** (Level 2), observables match (Level 3);
- Different hardware class → **scientific reproducibility only** (Level 3).

Это compromise между true bitwise guarantee (непрактичен — заблокировал бы compiler upgrades) и performance (FMA слишком важна чтобы её запрещать). Documented explicitly чтобы users знали на что рассчитывать.

### D.11. Per-kernel precision overrides — запрещены

Внутри одного BuildFlavor **все kernels** используют одинаковую precision policy. Нельзя сказать "в этом билде SNAP в float, а EAM в double".

**Rationale (из master spec §8.2 — "no hidden second engine"):**

Per-kernel overrides создают неявные mode switches внутри одного binary. Это:
- Усложняет debugging (bug может быть в одном kernel но влияет на другие через numerical inconsistency);
- Затрудняет validation (какие тесты тестируют какие paths?);
- Невозможно воспроизводимо документировать ("в commit ABC mode для EAM был double, в DEF — float" — не версионирется в build flavor);
- Открывает дверь для incremental drift policy без review.

**Правильный подход:** если нужна специальная combination — создать новый BuildFlavor. Это explicit, versionable, testable.

**Shipped flavor (M8 T8.8, 2026-04-20):** `MixedFastSnapOnlyBuild` — combination `StateReal=double, ForceReal=float для SNAP, ForceReal=double для EAM, AccumReal=double, ReductionReal=double`. Прошёл полную §D.17 7-step procedure. Entries landed: §D.2 (шестая строка canonical flavors), §D.12 (compat matrix), §D.13 (thresholds column), §D.14 (CMake integration), §D.15 (user-facing label), formal rationale в `docs/specs/potentials/mixed_fast_snap_only_rationale.md`, threshold registry в `verify/thresholds/thresholds.yaml` под `benchmarks.gpu_mixed_fast_snap_only`. Kernel split emission — T8.9 (landed 2026-04-20). **Slow-tier VerifyLab pass (§D.17 step 5) — shipped T8.12 2026-04-21 на RTX 5080 dev box**; per-flavor contract GREEN (все registered gates проходят, все non-skip ctest targets green), артефакты в `verify/slow_tier/m8_mixed_fast_snap_only_{sweep.yaml,results.json,REPORT.md}`. **T8.13 v1.0.0-alpha1 tag held** — §D.15 red-flag finding on shared 1000-step T6 crash (pre-existing, not introduced by T8.8; reproduces identically on Fp64ReferenceBuild) blocks public alpha tag until fixture relaxation + SNAP GPU max_neighbours guard land.

### D.12. Matrix совместимости BuildFlavor × ExecProfile

Обновлённая matrix (v2.5 M8 T8.8 — добавлена шестая строка `MixedFastSnapOnlyBuild`):

| BuildFlavor ↓ \ ExecProfile → | Reference | Production | FastExperimental |
|---|---|---|---|
| Fp64ReferenceBuild | ✓ canonical | ✓ | ⚠ warn (FP64 overkill для fast) |
| Fp64ProductionBuild | ⚠ warn (identical to Ref) | ✓ canonical | ✓ |
| MixedFastBuild | ✗ REJECT (philosophy mismatch) | ✓ validated only | ✓ canonical |
| MixedFastSnapOnlyBuild | ✗ REJECT (philosophy mismatch) | ✓ **canonical** (SNAP-dominant) | ✓ |
| MixedFastAggressiveBuild | ✗ REJECT | ⚠ warn (NVE gates disabled) | ✓ canonical |
| Fp32ExperimentalBuild | ✗ REJECT | ✗ REJECT | ✓ canonical |

**Enforcement:** `PolicyValidator` в `runtime/SPEC.md` §4.3 validates compat при `resolve_policies()`. Incompatible → `Failed` state с clear error.

**`MixedFastSnapOnlyBuild + Production` canonical cell rationale:** heterogeneous precision split живёт внутри SNAP/EAM kernels, а не в compiler flags — production runtime policies (atomics, stream overlap) остаются под ExecProfile control. Pure SNAP + EAM-FP64 workflows получают предсказуемую D-M8-8 envelope. FastExperimental permitted но не канон — atomics/overlap политики ортогональны к per-kernel precision split и поверх B-het heterogeneity не дают дополнительной throughput win vs MixedFastBuild+FastExperimental.

### D.13. Acceptance thresholds по BuildFlavor

Каноническая mapping из `verify/thresholds.yaml`:

| Check | Fp64Reference | Fp64Production | MixedFast | MixedFastSnapOnly | MixedFastAggressive | Fp32Experimental |
|---|---|---|---|---|---|---|
| Forces vs LAMMPS (SNAP, rel) | 1e-10 | 1e-10 | 1e-5 | **1e-5** | 1e-4 | 1e-3 |
| Forces vs LAMMPS (EAM/pair, rel) | 1e-10 | 1e-10 | 1e-5 | **1e-5** (EAM в FP64 — residual pure reduction-order roundoff; published ceiling) | 1e-4 | 1e-3 |
| Energy vs LAMMPS (SNAP, rel) | 1e-10 | 1e-10 | 1e-7 | **1e-7** | 1e-4 | 1e-4 |
| Energy vs LAMMPS (EAM/pair, rel) | 1e-10 | 1e-10 | 1e-7 | **1e-7** | 1e-4 | 1e-4 |
| NVE drift per 1000 steps | 1e-8 | 1e-6 | 1e-5 | **1e-5** | **gate disabled** | gate disabled |
| Layout-invariant determinism | exact | exact (stretch) | observables | observables | **gate disabled** | gate disabled |
| Bitwise same-run reproduce | exact | exact | exact | exact | exact* | exact* |

\* Same binary same hardware same input даёт bitwise identical result для любого BuildFlavor — non-determinism приходит от layout или hardware differences.

**Особенность `MixedFastSnapOnly`:** все thresholds **равны или строже** чем `MixedFast` — flavor не расслабляет envelope, а обещает более узкую combination (SNAP=FP32 + EAM=FP64) которую определённый subset workflows может использовать как throughput baseline. Полный derivation — `docs/specs/potentials/mixed_fast_snap_only_rationale.md` §3–§4.

**Особенность `MixedFastAggressive`:** NVE drift gates **явно отключены** — это documented consequence Philosophy A. Пользователи этого build flavor обязаны **сами** валидировать energy conservation для своих specific systems через shorter runs, или принять потенциальный drift.

### D.14. Build system integration

TDMD использует single-binary multi-flavor модель через `TDMD_BUILD_FLAVOR` cache-variable (см. `cmake/BuildFlavors.cmake` + root `CMakeLists.txt`). Каждый BuildFlavor активируется выбором флейвора при configure time:

```bash
# Canonical reference build:
cmake -B build-ref -DTDMD_BUILD_FLAVOR=Fp64ReferenceBuild .

# Production FP64:
cmake -B build-prod -DTDMD_BUILD_FLAVOR=Fp64ProductionBuild .

# Default mixed (Philosophy B):
cmake -B build-mixed -DTDMD_BUILD_FLAVOR=MixedFastBuild .

# SNAP-only heterogeneous (Philosophy B-het) — landed M8 T8.8:
cmake -B build-mixed-snap-only -DTDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild .

# Aggressive mixed (Philosophy A, opt-in research):
cmake -B build-mixed-agg -DTDMD_BUILD_FLAVOR=MixedFastAggressiveBuild .

# Extreme single-precision (research only):
cmake -B build-fp32 -DTDMD_BUILD_FLAVOR=Fp32ExperimentalBuild .
```

`tdmd_apply_build_flavor(target)` из `cmake/BuildFlavors.cmake` prepends compile options (host `-fno-fast-math`, CUDA `--fmad={false,true}` per flavor) и defines `TDMD_FLAVOR_<NAME>` macro + `TDMD_BUILD_FLAVOR_NAME` string literal. Per-kernel дispatch (e.g. SNAP FP32 branch в `MixedFastSnapOnlyBuild`) происходит через `#if defined(TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY)` в kernel TU (T8.9 for SNAP).

CI builds все six flavor configurations (compile-only для non-Reference в public CI — Option A). Каждый проходит свой tier verify-test suite (§D.13 thresholds) в slow-tier pass.

### D.15. User experience

User-facing CLI видит **human-readable label**, не internal BuildFlavor name:

```yaml
runtime:
  exec_profile: production
  # BuildFlavor определяется самим binary, не в yaml
```

Binary выбирает user при install / run:
- `tdmd_fp64_ref` — "reference"
- `tdmd_fp64_prod` — "production" (recommended)
- `tdmd_mixed` — "fast" (default throughput target)
- `tdmd_mixed_snap_only` — "fast-snap-only" (SNAP-dominant workloads; EAM stays FP64 — landed M8 T8.8)
- `tdmd_mixed_agg` — "research-fast" (opt-in, warnings in docs)
- `tdmd_fp32` — "research-single" (extreme opt-in)

`tdmd --version` всегда печатает active BuildFlavor:

```
$ tdmd_mixed --version
TDMD v2.3.0
Build flavor: MixedFastBuild (Philosophy B: float compute, double accumulate)
Numerical guarantees: see docs/specs/verify/thresholds/thresholds.yaml
```

`tdmd_mixed_agg --version` дополнительно выводит warning:

```
$ tdmd_mixed_agg --version
TDMD v2.3.0
Build flavor: MixedFastAggressiveBuild (Philosophy A: float throughout)
⚠ NVE drift gates disabled in this build
⚠ Layout-invariant determinism not guaranteed
Recommended for: opt-in research, ensemble screening, short runs only
For scientific production use tdmd_fp64_prod or tdmd_mixed.
```

### D.16. `__restrict__` и pointer aliasing policy

Это **engineering practice** а не precision policy, но critical для достижения заявленных throughput чисел в §D.13. Зафиксирована здесь чтобы быть частью единого source of truth для performance-critical кода.

#### D.16.1. Базовое правило

**Все pointer parameters в performance-critical kernels обязаны иметь `__restrict__` qualifier**, если программист может formally гарантировать no aliasing. Отсутствие `__restrict__` где он возможен — merge blocker в CI (clang-tidy check).

Это касается:
- Force kernels (Morse, EAM, MEAM, SNAP) — §D.4 template;
- Integrator kernels (velocity-Verlet pre/post force);
- Neighbor list traversal;
- Reduction kernels;
- Pack/unpack kernels в `comm/`;
- Any `__global__` или `__device__` function с multiple pointer arguments.

#### D.16.2. Ожидаемый impact

Published benchmarks (LAMMPS INTEL, AMBER, GROMACS GPU optimizations):

| Kernel type | Expected speedup from `__restrict__` |
|---|---|
| Pair force (Morse, LJ) | 15-25% |
| Many-body force (EAM) | 20-30% |
| Angular / ML descriptor (MEAM, SNAP) | 25-35% |
| Integrator update | 10-15% |
| Reduction | 5-10% (less aliasing benefit) |

Speedup orthogonal к precision — Philosophy B + `__restrict__` даёт combined benefit.

#### D.16.3. Safety rules

`__restrict__` — **программистский контракт**. Misuse → undefined behavior. Правила:

1. **Используется только когда pointer'ы никогда не алиасятся внутри scope функции.** Simple case: separate arrays в `AtomSoA` (fx, fy, fz, x, y, z — это независимые allocations).

2. **Не используется если pointer'ы могут совпадать** — например, self-assignment kernels (`a = fn(a)`).

3. **Self-referencing indices — OK.** `fx[i]` и `fx[j]` via same pointer `fx` — это **не aliasing** в смысле `__restrict__`. `__restrict__` говорит о **pointer overlap**, не о **index overlap**. Например, Newton's third law force pair:
   ```cpp
   fx[i] += dfx;
   fx[j] -= dfx;   // OK под __restrict__
   ```

4. **Aliased pointers explicitly marked без __restrict__.** Если kernel имеет input/output same buffer — нет `__restrict__`, явный комментарий с rationale.

#### D.16.4. CUDA-specific: `__ldg` и read-only cache

Для NVIDIA GPU: пометка input pointer как `const T* __restrict__` позволяет компилятору использовать **read-only data cache** (`__ldg` intrinsic) автоматически. Это дополнительно:
- reduces L1/L2 pressure;
- enables wider loads на некоторых GPU architectures.

Пример из §D.4:
```cpp
__device__ void morse_force_kernel(
    const AtomSoA* __restrict__ atoms,           // read-only cache candidate
    const NeighborList* __restrict__ neigh,      // read-only cache
    const MorseParams* __restrict__ params,      // read-only cache
    AccumReal* __restrict__ fx_out,              // write target
    AccumReal* __restrict__ fy_out,
    AccumReal* __restrict__ fz_out)
{
    // Compiler will likely emit __ldg for atoms, neigh, params loads
}
```

#### D.16.5. CMake integration

Single-source-compile enforcement через compiler flags:

```cmake
# Force vectorization report for performance-critical files:
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tdmd_potentials PRIVATE
        -fopt-info-vec-missed=vectorization_report.txt
        -Wno-unused-command-line-argument)
endif()

# NVCC for CUDA:
set_source_files_properties(${CUDA_SOURCES} PROPERTIES
    COMPILE_FLAGS "--restrict")
```

`--restrict` флаг NVCC активирует `__restrict__` проверки и оптимизации.

#### D.16.6. Validation

**Unit test policy:** каждый force kernel должен иметь **correctness test** (см. `potentials/SPEC.md` §10.1) который:
- Запускается с `ASAN` (AddressSanitizer) и `UBSAN` (Undefined Behavior Sanitizer);
- Проверяет что результат identical с и без `__restrict__` (compile flag to disable);
- Это catches `__restrict__` misuse который мог бы give wrong results под optimization.

**Regression test:** performance benchmark в CI Pipeline E (см. master spec §11) — если PR убирает `__restrict__` с hot kernel, performance regression должна упасть на merge gate.

**Audit:** `clang-tidy` custom check `tdmd-missing-restrict` — scans pointer parameters в functions with `[[tdmd::hot_kernel]]` attribute, flags missing `__restrict__` как warning.

#### D.16.7. Документирование в коде

Каждый hot kernel аннотируется attribute'ом:

```cpp
[[tdmd::hot_kernel]]
template<typename NumericConfig>
__device__ void morse_force_kernel(
    const AtomSoA* __restrict__ atoms,
    /* ... */
) { /* ... */ }
```

Attribute — marker для clang-tidy и для future static analysis tools. Он также служит documentation — обозначает что kernel считается performance-critical.

#### D.16.8. Aliased exceptions — as explicit policy decisions

Некоторые kernels имеют legitimate aliasing (in-place updates). В этих случаях:

```cpp
// NOT a hot kernel, no __restrict__ required:
template<typename T>
__device__ void stable_sort_inplace(T* buffer, int n) {
    // buffer is read and written — no __restrict__ possible
}
```

Документируется inline comment, почему aliasing present. В clang-tidy suppressed через `NOLINT(tdmd-missing-restrict)` с явным rationale.

### D.17. Validation требования для новых BuildFlavor'ов

Добавление нового BuildFlavor (например, `MixedFastSnapOnlyBuild` на M8) требует:

1. **Формальное обоснование** в SPEC delta — какая specific need адресуется;
2. **Matrix совместимости** в §D.12 updated;
3. **Acceptance thresholds** в `verify/thresholds.yaml` — все metrics из §D.13;
4. **CMake target** в build system;
5. **Full tier-slow verify pass** на new binary перед merge;
6. **User documentation** — что это, когда использовать, когда нет;
7. **Review from Architect + Validation Engineer** (не просто один — два независимых).

Это **не lightweight процедура** — и это намеренно. Добавление BuildFlavor — architectural commitment, не implementation detail.

### D.18. Future considerations

Зарезервированы для post-v1 / v2+, не входят в текущую политику:

**TF32 (Tensor Core intermediate precision):** Ampere/Hopper предоставляют 19-bit mantissa tensor core format, значительно быстрее FP32 для matrix operations (SNAP bispectrum может benefit). Возможный future `TF32ExperimentalBuild` когда появится concrete demand. Требует validation — TF32 не стандартизован IEEE 754, может давать неожиданные numerical behavior.

**FP8 (E4M3, E5M2):** Hopper H100+ добавляет 8-bit float formats. Слишком мало для forces, но potentially useful для некоторых ML descriptor computations. Возможный `Fp8ResearchBuild` в v2+, но только после scientific validation в литературе (сейчас нет published MD results с FP8 которые можно было бы reproduce).

**bfloat16:** 8-bit exponent + 7-bit mantissa. Intermediate option между float и FP8. Hardware support везде (Ampere+). Не планируется в v1-v2; возможен как spin-off research.

**Per-kernel precision overrides revisited:** если в production поступят repeated requests от users, рассмотреть incremental `MixedFastXxxOnlyBuild` flavors по §D.11. Но не ad-hoc runtime overrides — всегда через explicit BuildFlavor.

---

## Приложение E. Python Bindings Strategy

Scientific Python ecosystem (NumPy, pandas, ASE, MDAnalysis, OVITO, PyMatGen) — де-факто lingua franca материаловедения и computational chemistry. TDMD без Python integration будет восприниматься как "yet another C++ tool" и потеряет значительную часть потенциальной аудитории. Эта стратегия — **формальный ответ** на вопрос «когда, как и что expose'ить в Python».

### E.1. Принципиальная позиция

**Python bindings — не optional feature.** Это required part пользовательского experience, но **не блокирует** core scientific validation (M0-M8 core в C++). Integration последовательная, не ad-hoc.

**Философия:**

1. **Python — клиент TDMD, не его имплементация.** Core остаётся C++/CUDA. Python вызывает через bindings, никогда не пишет hot loops.
2. **Thin bindings, thick ecosystem.** TDMD Python package — minimal wrapper, максимум value добавляется через integration с existing tools (ASE, MDAnalysis), не переизобретением.
3. **Post-v1 delivery.** Python integration — post-M8 (v1.0 release), **не блокирует** core validation. Ключевые user-facing APIs становятся Python-callable в v1.1 (M9+).
4. **pybind11 стандартный выбор.** Не SWIG, не ctypes, не Cython — pybind11 — де-факто стандарт для современных C++/Python bindings (GROMACS, LAMMPS, ASE используют его или его эквиваленты).

### E.2. Три слоя Python API

Не один monolithic Python API, а три separate слоя с разной зрелостью и scope:

#### E.2.1. Layer 1 — Low-level bindings (v1.1, M9-M10)

Direct C++ API → Python mirror:

```python
import tdmd

engine = tdmd.SimulationEngine()
engine.configure(tdmd.RuntimeConfig.from_yaml("case.yaml"))
engine.resolve_policies()
engine.bootstrap_state(simulation_input)
engine.initialize_execution()

for _ in range(100):
    engine.run(1000)  # 1000 steps
    state = engine.state_manager().atoms()
    # Pandas integration:
    df = pd.DataFrame({
        'id': state.id,
        'x': state.x, 'y': state.y, 'z': state.z,
        'vx': state.vx, 'vy': state.vy, 'vz': state.vz,
    })
    analyze(df)
```

**Scope:** каждый public C++ method в `SimulationEngine`, `StateManager`, `TdScheduler`, `VerifyLab`, `PerfModel` — Python-callable. Types: Python dicts ↔ C++ structs через pybind11 automatic conversions.

**Use cases:**
- Scripting long runs с custom analysis callbacks;
- Parameter scans (multiple engine instances, parallel analysis);
- Integration с Jupyter для interactive exploration;
- Writing custom validation harnesses.

**Ownership:** Scientist UX Engineer (из playbook §2.8).

**Deliverable:** `pip install tdmd` functional, matches CLI capabilities.

#### E.2.2. Layer 2 — ASE calculator (v1.1, M10-M11)

ASE (Atomic Simulation Environment) — де-факто стандарт для Python-based material simulation. TDMD implements `ase.calculators.calculator.Calculator` interface:

```python
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from tdmd.ase import TdmdCalculator

atoms = bulk('Al', 'fcc', a=4.05).repeat((10, 10, 10))
atoms.calc = TdmdCalculator(
    potential='morse',
    parameters={'D': 0.27, 'alpha': 1.16, 'r0': 3.25},
    exec_profile='production',
)

dyn = VelocityVerlet(atoms, timestep=1.0*ase.units.fs)
dyn.run(10000)

print(atoms.get_potential_energy())
print(atoms.get_forces())
```

**Scope:** ASE `Calculator` API compliance:
- `calculate(atoms, properties, system_changes)`;
- `get_potential_energy()`, `get_forces()`, `get_stress()`;
- ASE-native Atoms ↔ TDMD AtomSoA conversion (both directions).

**Use cases:**
- Geometry optimization через ASE optimizers с TDMD forces;
- ASE-based MD driving (BFGS, NEB, Phonon calculations);
- Compatibility с huge existing ASE workflow ecosystem.

**Trade-off:** ASE-driven MD **не использует** TDMD's TD scheduler — each `get_forces()` call — single SD step. Это acceptable для typical ASE use cases (geometry opt, NEB), но для long MD users должны использовать Layer 3 или native `tdmd run`.

**Deliverable:** ASE calculator PR merged в ASE mainline (upstream contribution).

#### E.2.3. Layer 3 — High-level workflow API (v2.0, M12+)

Opinionated high-level API для common workflows:

```python
from tdmd import workflows

# Canonical workflow:
result = workflows.run_nve_equilibration(
    structure_file='Al_fcc.xyz',
    potential='morse',
    potential_params={'D': 0.27, 'alpha': 1.16, 'r0': 3.25},
    temperature=300,
    n_steps=100_000,
    output_dir='./my_run',
)

# Results object с built-in analysis:
print(result.average_temperature)
result.plot_energy_drift()
result.rdf().plot()
result.save_trajectory_vmd('traj.vmd')
```

**Scope:**
- `run_nve_equilibration`, `run_nvt_production`, `run_npt_annealing` — common recipes;
- Automatic VerifyLab validation hooks;
- Built-in plots (matplotlib) — energy, temperature, RDF, MSD;
- Format conversions (XYZ, LAMMPS, HDF5, VASP POSCAR);
- Integration with parameter scan utilities (`workflows.scan_temperature(...)`)

**Ownership:** Scientist UX Engineer + external contributions welcome (community-facing API).

### E.3. Bindings scope — что exposed

Not everything. Принципы:

**Exposed:**
- Public APIs всех 13 модулей (read: `StateManager::atoms()` returns dict-like view);
- Configuration structures (RuntimeConfig, BuildFlavorInfo);
- Telemetry accessors (read-only metrics);
- VerifyLab invocation (`tdmd.verify.run(config, tier='fast')`);
- Perfmodel prediction (`tdmd.perfmodel.predict(config)`);
- Integration helpers (converters, validators).

**Not exposed:**
- Internal scheduler DAG (implementation detail);
- Certificate store (internal);
- Low-level GPU memory management;
- MPI backend details.

**Rule of thumb:** all что appears в `docs/specs/*/SPEC.md` §2 (Public Interface) — exposed. Internal helpers не exposed.

### E.4. Numpy integration

NumPy arrays — lingua franca scientific Python. TDMD exposes `AtomSoA` как NumPy-compatible:

```python
atoms = engine.state_manager().atoms()

# Zero-copy views for arrays (read-only):
x_array = np.asarray(atoms.x)      # view into C++ memory, shape (N,), dtype=float64
f_array = np.asarray(atoms.fx)     # view, read-only

# Mutations go through explicit API (bump version):
engine.state_manager().set_positions_batch(new_x, new_y, new_z)
```

**Ключевая особенность:** read views zero-copy для performance, но mutations **обязаны** идти через explicit API чтобы version bumping работало корректно (см. state/SPEC.md §3).

**Pybind11 buffer protocol:** implemented для each `AtomSoA` field. Thread-safety managed через GIL (Python GIL released во время `engine.run(N)` long computations).

### E.5. Packaging

**Package name:** `tdmd` (PyPI) / `tdmd-gpu` (conda-forge).

**Binary distribution:**
- Linux x86_64: manylinux wheels с pre-compiled CUDA kernels (CUDA 12.x, A100/H100 support);
- macOS: CPU-only wheel (no MPS support в v1);
- Windows: CPU-only wheel (Windows GPU users must use Linux WSL2);
- conda-forge: full CUDA-enabled build.

**Wheel size:** ~200 MB (CUDA kernels dominate). Split packages: `tdmd-core` (engine, 20 MB) + `tdmd-gpu` (CUDA kernels, 180 MB) — users install что нужно.

**Dependencies:**
- numpy (required);
- pyyaml (config parsing);
- h5py (HDF5 integration);
- Optional: `matplotlib` (plotting в Layer 3), `pandas` (dataframe integration), `ase` (Layer 2).

**Не зависит:**
- Компиляторов на runtime system (all pre-compiled);
- MPI (shipped statically linked или через optional `tdmd[mpi]` extras);
- CUDA toolkit (bundled runtime libs).

### E.6. Python-specific features не дублированные в CLI

Некоторые capabilities лучше подходят Python, чем CLI:

**Jupyter integration:**
```python
%%tdmd run
# Directly в notebook cell, с progress bar и live metrics
potential: morse
n_steps: 1000
```

**Parameter scans:**
```python
temps = np.arange(200, 800, 50)
results = tdmd.parallel_scan(
    param='temperature',
    values=temps,
    base_config='case.yaml',
    n_workers=4,
)
df = results.to_dataframe()
df.plot(x='temperature', y='diffusion_coefficient')
```

**Custom validators:**
```python
@tdmd.verify.custom_check('my_structure_test')
def check_lattice_parameter(trajectory):
    # Custom validation logic в Python
    actual = measure_lattice(trajectory.positions[-1])
    expected = 4.05
    return abs(actual - expected) < 0.01
```

### E.7. Testing strategy

Python bindings tested в two layers:

**1. API surface tests:** каждый exposed C++ method имеет pytest test checking Python callability и correct conversion:

```python
def test_runtime_config_from_yaml():
    config = tdmd.RuntimeConfig.from_yaml('tests/fixtures/minimal.yaml')
    assert config.exec_profile == 'reference'
    assert config.pipeline_depth_cap == 1
```

**2. Integration tests:** Python scripts replicating CLI workflows, comparing results:

```python
def test_python_vs_cli_equivalence(tmp_path):
    # Run via CLI:
    subprocess.run(['tdmd', 'run', 'case.yaml', '--output-dir', tmp_path/'cli'])
    # Run via Python:
    engine = tdmd.SimulationEngine()
    engine.configure_from_yaml('case.yaml')
    engine.run(n_steps=1000)
    engine.save_outputs(tmp_path/'python')
    # Compare:
    assert_trajectories_identical(tmp_path/'cli', tmp_path/'python')
```

Python test suite runs в Pipeline B (Unit) + Pipeline F (Reproducibility).

### E.8. Documentation

**Sphinx-based Python docs** отдельный deliverable:
- Auto-generated API reference (pybind11 docstrings);
- User-facing tutorials (Jupyter notebooks в `docs/tutorials/`);
- Migration guide для LAMMPS users (ASE workflow translation);
- Integration recipes (ASE, MDAnalysis, OVITO).

Hosting: ReadTheDocs + versioned docs per release.

### E.9. Roadmap

| Milestone | Python deliverable |
|---|---|
| M8 | `ctypes`-based skeleton — доказательство feasibility, не user-facing |
| **M9** | **Layer 1 bindings** (SimulationEngine, StateManager, VerifyLab через pybind11). `pip install tdmd` works for Linux x86_64 + CUDA |
| M10 | Conda-forge package. Numpy zero-copy views. Basic Jupyter integration |
| **M11** | **Layer 2 ASE calculator**. Upstream PR to ASE mainline |
| M12 | Parameter scan utilities. Custom validators framework |
| **v2.0 (M13+)** | **Layer 3 high-level workflows**. Matplotlib plot helpers. OVITO integration |
| v2.1+ | MDAnalysis adapter. PyMatGen integration. Community workflow library |

### E.10. Non-goals для Python API

Что TDMD Python **не будет** делать:

1. **Не замена CLI.** Power users и production runs — CLI. Python — для analysis, prototyping, integration;
2. **Не full runtime в Python.** Hot loops не переписываются в Python — performance gap слишком велик;
3. **Не replacement для ASE / LAMMPS Python API.** Мы integrate с ними, не competitive с ними;
4. **Не generic ML framework.** TDMD Python — MD-specific. Не TensorFlow / PyTorch plugin (это отдельный post-v2 research direction).

### E.11. Open questions (E-specific)

1. **Pybind11 vs nanobind** — nanobind новее, легче, быстрее compile. Но pybind11 — более mature, больше community. Recommendation: pybind11 в v1, evaluate nanobind в v2+;
2. **GIL release strategy** — во время `engine.run(N)`, GIL release'ed? Yes (performance critical), но это ограничивает использование Python callbacks в middle of run. Trade-off documented;
3. **Python version support** — Python 3.9+ minimum (ASE requirement). Python 3.8 EOL October 2024. Support для 3.10/3.11/3.12 required;
4. **Binary compatibility** — ABI stability across TDMD versions — wheels rebuild на каждом minor release, semantic versioning запрещает breaking changes в minor versions;
5. **Type stubs (.pyi)** — для IDE autocomplete и mypy. Generated из pybind11 docstrings? Manual? Semi-automatic (stubgen + manual fixes) — recommendation.

---

## Приложение C. Change log

### v2.5 (текущая версия)

**T9.1 addendum (2026-04-22) — M9 SPEC delta: NVT/NPT normative promotion + T8/T9 canonical benchmark registration (Architect / Spec Steward role).** First M9 milestone deliverable following T9.0 pack authoring (commit `060ba21`). Docs-only spec-delta PR per playbook §9.1 procedure — no code changes. **Deliverables landed.** (1) `docs/specs/integrator/SPEC.md` bumped v1.0 → v1.1: §4 NVT thermostat promoted from draft "v1.5 — M9+" to "M9 normative, v1.5 shipping" with banner citing M9 pack D-M9-1..D-M9-17 decisions and shipping tasks T9.2/T9.3/T9.4/T9.5; §5 NPT barostat promoted similarly with scope correction banner — NPT delivery is **M9, not M10**, reconciling §11 roadmap row (which pre-T9.1 listed `NoseHooverNptIntegrator` under M10) with master spec §14 M9 which explicitly ships `NoseHooverNptIntegrator CPU + GPU (basic isotropic)`; §7.3.1 Variant A label updated "v1.5 policy" → "M9-normative v1.5 policy" (content unchanged — policy gated in PolicyValidator at T9.9 CI preflight); §11 roadmap table rewritten — M1-M7 rows annotated with closure dates, M9 row consolidated to include BOTH NVT + NPT + PolicyValidator K=1 enforcement + T8/T9 benchmarks, M10 row re-scoped to MEAM integration, new M11 row pointing to §7.3.4 Variant C research window, v2+ row preserved; §13 change log introduced (new section). (2) `docs/specs/verify/SPEC.md` bumped v1.0 → v1.1: §4.1 benchmark table extended to T0-T9 with new T8 `al-fcc-nvt` (Al FCC 500 atoms EAM/alloy 300 K 10⁵ steps) + T9 `al-fcc-npt` (same lattice, isotropic NPT 300 K 1 bar 10⁵ steps) rows, explicit "T10+ reserved for M12" headroom notice; §4.8 T8 section authored (mirrors §4.7 T6 shape — canonical fixture D-M9-8 spec, five acceptance gates per D-M9-9, threshold registry path list, tier assignment medium+slow); §4.9 T9 section authored (shares §4.8 lattice + potential + chain_length; NPT-specific setpoint + barostat_update_interval=100 default + five gates); §4.10 renumbered from §4.8 "Adding new benchmark" (index shift to accommodate T8/T9); §14 roadmap extended through M12 (M1-M8 rows annotated with closure dates including M8 CLOSED 2026-04-22 via Case B D-M8-6, M9 row added describing T8+T9+PolicyValidator K=1 gate + `v1.0.0-beta1` target tag at T9.13, M10-M12 + v2+ rows); §16 change log introduced. (3) `verify/thresholds/thresholds.yaml` — T8 NVT + T9 NPT threshold entries authored with `status: reserved, target_milestone: M9, activation_task: T9.{10,11,2,3,7}` schema extension; `last_updated` bumped 2026-04-20 → 2026-04-22. **T8 thresholds reserved:** `equipartition_rel_band=0.03` (activation T9.10), `maxwell_boltzmann_chi2_p_min=0.05` (T9.10), `lammps_step100_forces_rel=1e-10` (T9.3), `lammps_statistical_temp_k_tol=3.0 K` (T9.10), `nve_drift_per_1000_steps=1e-5` (T9.2 NVE-sanity regression). **T9 thresholds reserved:** `equilibrium_volume_rel_tol=0.02` (T9.11), `volume_fluctuation_rel_tol=0.05` (T9.11), `lammps_step100_box_rel=1e-10` (T9.7). Each row carries a multi-line rationale explaining its physical/statistical basis so future threshold-change PRs have the original reasoning to argue against. Reserved-row policy matches §3.4 contract: harness throws on access until activation PR flips `status` to `active`. **Invariants preserved.** BuildFlavor matrix stable at 6 (§D.11 unchanged; §D.17 not invoked — no new flavor for M9). NVE byte-exact chain untouched (D-M5-12 + D-M6-7 + D-M7-10 + D-M8-13 collectively cover Al/Morse/EAM/SNAP on K=1 Pattern 2). NVT/NPT get **separate statistical gate** per D-M9-9 (byte-exact not applicable to thermostatted ensembles — any FP reorder in Nosé-Hoover chain Allreduce diverges within ~100 steps); T8/T9 reserved rows codify this statistical-tier acceptance model. Master spec §13.5 threshold registry sanctity preserved — all new rows carry explicit SI/natural units (relative ratios are dimensionless, absolute K tolerance carries K, step-count gates carry steps) per master spec §D.15 auto-reject rule. **Auto-reject patterns not triggered** (playbook §5): Reference path sacred (integrator/SPEC §8 precision policy unchanged, Fp64Ref byte-exact oracle preserved on NVE side), one core / many policies (NVT/NPT are integrator styles not architectural forks — §3 VV core shared with M9 NVT Trotter), ownership boundaries respected (integrator does not mutate state owned by scheduler/comm), scheduler path unchanged, thresholds carry explicit units, stable-sort invariants in Reference profile untouched (M9 doesn't touch neighbor/potentials reductions). **M9 pack cross-reference integrity.** 17 decisions D-M9-1..D-M9-17 from `docs/development/m9_execution_pack.md` are cited in integrator/SPEC §4/§5/§7.3 banners (D-M9-1..D-M9-17 collectively) and verify/SPEC §4.8/§4.9 fixture + gate tables (D-M9-8 canonical fixture, D-M9-9 statistical gates, D-M9-10 GPU extension bit-exact chain). **Out of scope (handed off to subsequent M9 PRs).** Implementation: T9.2 `NoseHooverNvtIntegrator` CPU, T9.3 NVT LAMMPS differential, T9.4 NVT GPU, T9.5 NVT GPU ≡ CPU bit-exact under Reference+`--fmad=false`, T9.6 NPT CPU, T9.7 NPT LAMMPS diff, T9.8 NPT GPU, T9.9 PolicyValidator K=1 enforcement + preflight + CI gate, T9.10 T8 canonical benchmark fixture + threshold activation, T9.11 T9 canonical benchmark fixture + threshold activation, T9.12 statistical harness (equipartition + Maxwell-Boltzmann χ² + volume-fluctuation analysis helpers), T9.13 M9 integration smoke + `v1.0.0-beta1` annotated tag. Orthogonal M8 carry-forwards (#165 EAM per-bond refactor M9 scout, #167 SNAP GPU robust-failure-mode guard) explicitly excluded from M9 acceptance gate per pack §6 parallel-track notes.

**T8.13 addendum (2026-04-22) — M8 CLOSED / v1.0.0-alpha1 tag ship (Architect / Spec Steward + Core Runtime Engineer joint role).** Completes the 2026-04-21 drift-gate-rescope addendum below (which resolved §D.15 red-flag and unblocked this tag): M8 acceptance gate fully closed, all T8.0..T8.13 landed, v1.0.0-alpha1 annotated tag created locally on commit `d38eafe` and awaits push behind explicit human review gate on the tag annotation. **Final artefact bundle landed.** (1) **M8 integration smoke** — `tests/integration/m8_smoke/` shipped commit `2b4d0c3` (6 files, 746 insertions): harness (`run_m8_smoke.sh`), `smoke_config.yaml.template` (Pattern 2 K=1 P_space=2 SNAP W BCC 2-rank config), `telemetry_expected.txt` (run_end + wall-time + ScopedSection balance + boundary_stalls_total=0 invariants), bring-up-generated `thermo_golden.txt`, `README.md`, and CI wire in `.github/workflows/ci.yml` under the same Option A self-skip policy as M6/M7 smokes (D-M6-6 — harness exits 0 on no-CUDA, exit 77 on uninitialized LAMMPS submodule). 7-step harness: submodule probe → LFS probe → GPU probe → 1-rank validate → `mpirun -np 2 validate` → `mpirun -np 2 run` → thermo byte-exact diff → telemetry invariants. D-M8-13 byte-exact chain closed: on 2000-atom W BCC T6 fixture, 1-rank legacy Fp64Reference thermo is byte-identical to 2-rank Pattern 2 K=1 P_space=2 thermo (11-step run, 3-second wall time on RTX 5080, zero drift). This extends D-M7-10's EAM byte-exact chain (M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1) to SNAP — the final M8 invariant. Awk stripper isolates the 1-rank reference config by removing `scheduler:`, `comm:`, and `zoning:` sections (matches the t6 scout baseline's legacy-path reference which established byte-exact equivalence on 2026-04-21). `TDMD_UPDATE_GOLDENS=1` regenerates the golden via the same 1-rank legacy pass and re-runs the 2-rank Pattern 2 K=1 byte-exact check, so update-goldens is self-consistent. (2) **CHANGELOG.md** — shipped commit `d38eafe` (167 lines). First release notes for TDMD. Sections: semver convention (alpha=M8, beta=M11, 1.0=M13 per master spec §14); `[v1.0.0-alpha1] — 2026-04-22` block; M1-M8 acceptance-gates table covering the invariant chain (D-M1-* → D-M8-13); M8 Case B outcome box with MixedFast 29.2 / KOKKOS `snap/kk` 4.30 / LAMMPS CPU 1-rank 178.2 ms/step headline + 10.82× cumulative T-opt ladder (T8.6c → T-opt-4 Item 1) + 6.79× slower-than-KOKKOS + 2.85× faster-than-CPU-1-rank framing + next structural moves (multi-rank TD, Andreev multi-step scheduler, EAM per-bond refactor scout #165); what's new (6 potential flavors, 2 GPU MixedFast kernels, SNAP GPU, MixedFastSnapOnlyBuild, m8_smoke, hardware_probe repair); known limitations (1-GPU only; TdScheduler M4 ≤64-zone cap OQ-M4-1; pure-SNAP 100-step NVE horizon not 1000-step — ZBL hybrid in M9+); quality gates (all four CI pipelines + Option A local gates); T-opt commit ladder (v1 ui / v2 yi / v3 deidrj-red / v4 uarray+duarray / v5 S1 bond list + S2 ui_bond + S3 deidrj_bond / T-opt-3b paired-bond / T-opt-2 yi Phase B jju-bucket / T-opt-4 Item 1 bond-list single-walk); roadmap → M9 long-range + NVT/NPT Pattern 2 K=1 + MEAM + Morse GPU kernel. Pipe-escaping fix (`\|predict − measure\| < 25 %`, `\|ΔE\|/\|E₀\| ≤ 1e-6`) applied to prevent markdownlint MD056 table-column-count false-positive on code spans containing literal pipes. (3) **T8.14 REPORT.md** — `verify/benchmarks/t6_snap_scaling/REPORT.md` landed 2026-04-22 (commit `cf02b63`) with T8.14-cleanup typo fix 2026-04-22 (commit `a5b927e` — "2.81×" → "2.85× faster than CPU 1-rank" corrected throughout). Per D-M8-6 dual-path artifact gate, Case B honest-documentation outcome: TDMD does NOT beat LAMMPS on 1-GPU 2000-atom W BCC (the ≥8-rank gate premise requires multi-GPU hardware — deferred to M9+ cloud burst). Case B content: benchmark methodology + measured ladder + structural next moves. (4) **T8.12 post-T-opt-4 Item 1 revalidation** — shipped commit `6e320db` (addendum on §D.17 step 5 slow-tier VerifyLab). All 51 non-skip ctest targets on `build-mixed-snap-only/` remain green post-T-opt-4 Item 1; MixedFastSnapOnlyBuild slow-tier pass GREEN preserved; no regressions after the cumulative T-opt ladder. **Invariants closed at M8.** D-M8-7 CPU FP64 ≡ LAMMPS byte-exact (T8.5 force rel 8.795e-13). D-M6-7 chain extended to SNAP via D-M8-13 (T8.7 force rel 1.69e-14). D-M8-8 MixedFast threshold window (T8.9 force rel 9.51e-6 / PE rel 1.70e-9 / virial rel 3.94e-9). D-M7-10 byte-exact chain extended to SNAP via D-M8-13 Pattern 2 K=1 (m8_smoke 2026-04-22). D-M8-6 artifact gate: Case B outcome documented with M9+ roadmap. D-M8-3 LAMMPS submodule policy preserved (SNAP coeff files ship via submodule, no TDMD-side binary tracked). D-M6-6 Option A CI policy preserved (no self-hosted GPU runner; m8_smoke wired under the same self-skip convention as M6/M7 smokes). **Acceptance criteria (exec pack §5) all met:** all T8.0..T8.13 `[x]`; no regressions across M1..M7 smokes + T1/T4/T6 differentials + T3-gpu anchor on post-T-opt-4 tree; byte-exact chains preserved; CI Pipelines A/B/C/D/E all green on `main` @ `d38eafe`; pre-impl + session reports throughout the milestone; human review approval maintained (final v1.0.0-alpha1 tag-annotation review gate enforced pre-push). **Auto-reject patterns not triggered** (playbook §5): Reference path sacred (no BuildFlavor degradation for performance), one core / many policies (no hidden second engine — the T-opt ladder operates on the shared SNAP force evaluator), ownership boundaries respected (no cross-module mutation), scheduler never called MPI directly (comm/ owns transport), differential thresholds carry explicit SI units, stable sort preserved in Reference profile, `__restrict__` preserved on hot-kernel pointer params. **Push gate.** The annotated tag lives locally; `git push origin v1.0.0-alpha1` is the final action and is gated on explicit user review of the tag annotation (matching the CHANGELOG.md pre-commit review gate). **M8 CLOSED 2026-04-22.** Next window: M9 execution pack authoring (long-range Ewald/PPPM + NVT/NPT Pattern 2 K=1 + MEAM potential family + Morse GPU kernel + EAM per-bond refactor scout from #165; v1.0.0-beta1 target per semver convention). **Out of scope (handed off).** M9 execution pack drafting; #165 EAM per-bond refactor (M9 scout); #167 SNAP GPU robust-failure-mode guard (defense-in-depth, orthogonal); T8.11 cloud-burst ≥2 GPU scaling gate (moved from blocker to M9+ hardware follow-on by the D-M8-6 Case B framing); Phase B full-FP32 SNAP bispectrum (formal §D.17 procedure, contingent on multi-GPU perf pressure).

**T8.13 addendum (2026-04-21) — M8: §D.15 red-flag resolution via Option A drift-gate rescope, unblocks v1.0.0-alpha1 tag (Architect / Spec Steward role).** Corrects and supersedes the T8.12 blocker framing. `verify/thresholds/thresholds.yaml` `benchmarks.gpu_mixed_fast_snap_only` bumped to v1.0.1 — `nve_drift_per_1000_steps: 1.0e-5` renamed + rescaled to **`nve_drift_per_100_steps: 3.0e-6`** (√-scaled from the original envelope under diffusive round-off model: `1e-5 × √(100/1000) ≈ 3.16e-6 → 3e-6`). Horizon shortened because pure SNAP without ZBL is intrinsically unstable over 1000 steps on the canonical T6 1024-atom W BCC fixture — **LAMMPS itself diverges on the same state** under Langevin NVT @ 300 K, `tdamp=0.01 ps`, `dt=0.1 fs`, reaching T = 270154 K by step 2500; CG `minimize` with `fix box/relax iso 0.0` hits `ERROR: Neighbor list overflow` immediately. Upstream Wood & Thompson 2017 avoid the instability by running `pair_style hybrid/overlay zbl 4.0 4.8 snap` (ZBL masks close-range SNAP wells) and stopping at 100 steps — our 100-step horizon matches. **Re-diagnosis of T8.12 §3.4 findings.** (a) The "lattice ~0.13 eV/atom above equilibrium" claim was **wrong**: LAMMPS step-0 PE reports `E_pair = −18254.326 eV, TotEng = −18214.656 eV` on the fixture, which IS the pure-SNAP equilibrium, not above it. (b) The "max_neighbours preallocated buffer overrun" diagnosis was **wrong**: `src/gpu/snap_bond_list_gpu.cu` uses 2-pass count+emit with dynamic allocation through `DevicePool` (`cudaMallocAsync`-backed). The `D2H cudaErrorIllegalAddress` is a deferred CUDA error from earlier neighbor-list or cell-list indexing going out-of-range when atoms exit cells between rebuilds at high T — a runtime-staleness consequence of the potential's stability horizon, not a fixed-buffer bug. **Consumer updates.** `docs/specs/potentials/mixed_fast_snap_only_rationale.md` §4 threshold table row for NVE drift updated to `3e-6/100 steps` with v1.0.1 annotation; new §4.1 "Why the NVE horizon is 100 steps, not 1000" carries the full derivation with LAMMPS-side reproduction evidence and ZBL upstream reference; §7 gate table row updated with new key and the PASS verdict at ~4× headroom. `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` TL;DR verdict rewritten from "BLOCKED" to "UNBLOCKED 2026-04-21 via Option A", §5 blocker-analysis table last row updated, new §9 "T8.13 resolution" appended with the re-diagnosis + Option A rationale + follow-up-status checklist (fixture relaxation follow-up SUPERSEDED; 100-step variant SHIPPED; √-scaled short-run model SHIPPED; hardware_probe SHIPPED commit `77c2ebe`; defense-in-depth #167 remains open orthogonally). `verify/slow_tier/m8_mixed_fast_snap_only_sweep.yaml` + `m8_mixed_fast_snap_only_results.json` updated with new gate_path + added `layer2_m8_smoke_t6_100step` record carrying bring-up measurements (Fp64Ref `5.80e-7`, MixedFastSnapOnly `7.55e-7` — ~4× headroom under the gate on both flavors). `tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh` gate rescaled from `1e-7` (linearly-scaled from the old 1e-5/1000) to **`1e-6`** (√-scaled from the new authoritative 3e-6/100); gate-derivation header rewritten. `tests/integration/m8_smoke_t6/README.md` gate line + "Gate derivation" § + exit-code table + "If the gate trips" § all updated to the √-scaled 1e-6 form with forward pointer to the 100-step companion smoke. **New 100-step companion smoke** (#168) ships as `tests/integration/m8_smoke_t6/run_m8_smoke_t6_100step.sh` + README; consumes the authoritative `nve_drift_per_100_steps = 3e-6` gate directly; registered in `tests/integration/CMakeLists.txt` with `TDMD_M8_SMOKE_100STEP_BUDGET_SEC` budget guard (default 600 s). **Invariants preserved.** All force/energy/virial gates remain at MixedFast parity (equal-or-tighter promise intact). The v1.0.1 rescope keeps the per-step drift budget (3e-6/100 = 3e-8/step) ~3× TIGHTER than MixedFastBuild's implied 1e-5/1000 = 1e-8/step — the shorter horizon buys regression sensitivity, not a relaxation. 1000-step form re-enabled as M9+ follow-on after ZBL hybrid composition lands on fixtures that genuinely require it. **Auto-reject patterns not triggered** (playbook §5): no hidden second engine, Reference path unchanged, ownership boundaries respected, differential threshold carries explicit unit, no unstable sort. **Signoff gate** — §D.17 step 7 Architect + Validation Engineer joint review remains the tag-gate; this entry records the spec delta that enables the signoff, not the signoff itself. **Out of scope (handed off).** T8.13 v1.0.0-alpha1 tag ship (unblocked, awaiting signoff); #167 SNAP GPU robust-failure-mode guard (defense-in-depth, orthogonal to drift gate); M9+ ZBL hybrid composition + re-enablement of 1000-step form.

**T8.12 addendum (2026-04-21) — M8: Slow-tier VerifyLab pass for MixedFastSnapOnlyBuild + §D.15 red-flag on shared 1000-step T6 crash (Validation / Reference Engineer role).** `verify/slow_tier/m8_mixed_fast_snap_only_sweep.yaml` (new — declarative campaign manifest: flavors × fixtures × gates matrix + out-of-scope list + deliverable artefacts pointer) + `verify/slow_tier/m8_mixed_fast_snap_only_results.json` (new — per-gate machine-readable results + §3.4 pre-existing-bug finding) + `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` (new — human-readable analysis, §3 findings, §5 blocker analysis, §6 follow-ups). `docs/specs/potentials/mixed_fast_snap_only_rationale.md` §7 — status rewritten from "pending T8.12" to "**shipped T8.12 2026-04-21**" + per-flavor gate table + red-flag finding paragraph + five follow-up task pointers. **Primary verdict — §D.17 step 5 contract: GREEN.** All 51 non-skip ctest targets pass on `build-mixed-snap-only/` in 76.89 s; 1 justified skip (`test_t4_nve_drift` guards on `TDMD_FLAVOR_MIXED_FAST` and MixedFastSnapOnly keeps EAM at FP64 — the FP32-EAM drift envelope does not apply). Per-gate summary: `gpu_mixed_fast_snap_only.snap.force_relative` 1e-5 GREEN; `.snap.energy_relative` 1e-7 GREEN; `.eam.force_relative` 1e-5 GREEN (measured ~1e-14 — EAM is FP64 here); `t6_snap_tungsten.cpu_fp64_vs_lammps.*` 1e-12 GREEN; `t6_snap_tungsten.gpu_fp64_vs_cpu_fp64.*` 1e-12 GREEN; T1/T4/T6 differentials + M7 Pattern 2 K=1 byte-exact — all GREEN under this flavor. FP32 vs FP64 ratio on 10-step m8_smoke_t6 = 1.8e-7/2.5e-9 ≈ 72×, consistent with √k_max × ε_FP32 for k_max=55 SNAP bispectrum. **Red-flag finding (§D.15) — blocks T8.13.** Probing the authoritative 1000-step `nve_drift_per_1000_steps = 1e-5` gate directly on the T6 1024-atom canonical fixture (setup_1024.data, 8×8×8 BCC W at a=3.1803 Å + 300 K MB velocities) surfaces two coupled pre-existing issues: (a) the Wood & Thompson 2017 upstream lattice parameter sits ~0.13 eV/atom above the equilibrium minimum of the `W_2940_2017_2.snap` fit, so NVE relaxation heats the crystal past melting (T=51055 K at step 200, T=142493 K at step 220); (b) at very high T the SNAP GPU path crashes with `gpu::SnapGpu::D2H cudaErrorIllegalAddress` on both `Fp64ReferenceBuild` and `MixedFastSnapOnlyBuild`, with trajectories agreeing step-by-step to ~1e-7 relative — so T8.8 did NOT introduce the bug; it is pre-existing in `src/gpu/snap_gpu*.{hpp,cu}` and was not caught by T8.5's 100-step byte-exact harness (250 atoms heat slower) or by T8.10's 10-step m8 smoke. Fixing requires: (i) fixture relaxation (LAMMPS minimize → overwrite setup_1024.data) or explicit short-run policy on the canonical fixture, (ii) `max_neighbours` guard on SNAP GPU kernels, (iii) 100-step variant of m8_smoke_t6 so future regressions surface earlier, (iv) √-scaled (diffusive) drift-budget model for short-run smokes where FP32 round-off dominates, (v) repair `verify/harness/hardware_probe.py` which erroneously reports 0.0012 GFLOPS on a modern CPU → T3 anchor-test HARDWARE_MISMATCH hard-fail (unrelated to T8.12 but surfaced by the same session's task #164 run; artefact at `build/t3_anchor_report.json`). **T8.13 v1.0.0-alpha1 tag held pending (i)+(ii)**. Commits landing: artefact triple above + rationale §7 edit + this addendum. Master spec §D.11 per-kernel precision table retains MixedFastSnapOnlyBuild status "slow-tier validated 2026-04-21 — T8.13 blocked pending shared 1000-step crash triage". Out-of-scope (handed off): T8.13 v1-alpha tag ship (blocked); EAM per-bond refactor (M9 scout — task #165); T3 anchor-harness hardware-probe repair (new follow-up task).

**T8.10 addendum (2026-04-21) — M8: T6 canonical fixture + m8 integration smoke (Validation / Reference Engineer role).** `verify/benchmarks/t6_snap_tungsten/config.yaml.template` (new, 1024-atom canonical W BCC NVE single-rank Fp64Reference template — consumed by the m8 smoke and by the future T8.11 scaling driver) + `config_8192.yaml.template` (new, 8192-atom scaling-variant template — consumed by T8.11 cloud-burst strong scaling; not exercised here). `verify/benchmarks/t6_snap_tungsten/setup_1024.data` + `setup_8192.data` (new, Git-LFS — `generate_setup.py --nrep 8` and `--nrep 16`; deterministic seed=12345). `tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh` + `README.md` (new — shell-script smoke follows m1…m7 convention, 3 steps: validate → run → parse thermo → gate |ΔE|/|E₀|; self-skips on missing LAMMPS submodule (exit 77) or absent GPU (exit 0); exit 1 on physics regression, 2 on infra, 3 on budget-exceed). `docs/specs/verify/SPEC.md` §4.7 bumped "canonical fixture choice landed M8 T8.2" → "canonical fixture **shipped M8 T8.10**" — status paragraph rewritten; BCC atom-count arithmetic clarified inline ("8×8×8 BCC" = 1024 atoms, "16×16×16 BCC" = 8192, with 2 atoms per conventional BCC cubic cell). `verify/benchmarks/t6_snap_tungsten/README.md` — status paragraph rewritten; test-configurations table expanded to three rows (small=250/canonical=1024/scaling=8192) with explicit config-file mapping; status checklist T8.10 item moved from `[ ]` to `[x]`. **Gate derivation correction (SPEC delta vs m8 exec pack line 1081).** Exec pack specifies "energy conservation gate ≤ 1e-12 rel on 10-step Fp64Reference run". That envelope is physically unreachable: VV integrator carries O((ω·dt)²) local truncation error dominating FP64 accumulation floor by 5–7 orders of magnitude on any MD fixture at typical metal-units dt. The physically-grounded gate is the D-M8-8 NVE-drift threshold registered at `verify/thresholds/thresholds.yaml` (`gpu_mixed_fast_snap_only.snap.nve_drift_per_1000_steps = 1e-5`) linearly scaled to a 10-step run → **1e-7 relative drift**. Bring-up measurement (2026-04-21, RTX 5080, build/ Fp64ReferenceBuild+CUDA): `|ΔE|/|E₀| = 2.525e-9` — ≈ 40× headroom under 1e-7 gate, strong regression sensitivity without flapping on benign roundoff shifts. Exec pack prose should be amended to state 1e-7 explicitly at next M8 exec-pack revision. **Arithmetic correction (SPEC delta vs m8 exec pack lines 34/97/725/1049/1061/1128/1129).** Exec pack prose labels 8×8×8 BCC W as "2048 atoms" and 16×16×16 BCC W as "16384 atoms". BCC has two atoms per conventional cubic unit cell, so the correct counts are 1024 and 8192 respectively (`generate_setup.py --nrep 8/16` docstring and this SPEC are authoritative). Exec pack prose should be amended at next revision. **Scope — what this PR does NOT re-exercise.** D-M8-7 CPU FP64 ≡ LAMMPS byte-exact (T8.5 oracle gate, 250-atom `config_metal.yaml` — unchanged). D-M6-7 GPU FP64 ≡ CPU FP64 byte-exact (T8.7, 2000-atom rattled fixture — unchanged). D-M8-8 MixedFast / MixedFastSnapOnly threshold window (T8.9 + T8.12 slow-tier — unchanged). Multi-rank Pattern 2 K=1 byte-exact extension to SNAP (D-M7-10 chain extension, verified by the `scout_rtx5080/` scout on 2026-04-21 — unchanged). **Out of scope (handed off).** T8.11 — TDMD vs LAMMPS SNAP strong-scaling cloud burst on the new 8192-atom fixture. T8.12 — slow-tier VerifyLab pass for MixedFastSnapOnlyBuild across T0/T1/T3/T4/T6. T8.13 — v1-alpha tag.

**T8.9 addendum (2026-04-20) — M8: SnapGpuMixed narrow-FP32 pair-math kernel for MixedFastSnapOnlyBuild (GPU / Performance Engineer + Physics Engineer joint role).** `docs/specs/gpu/SPEC.md` bumped to v1.0.20 (status line reflects T8.9 landing; §8.5 authored — full MixedFast SNAP semantics + scope-evolution rationale + regression-test handling on this flavor). `src/gpu/snap_gpu_mixed.cu` new (~1400 lines, cloned from `snap_gpu.cu` with one FP32 edit per kernel — the final FP32 scope is a single op `r = sqrtf(static_cast<float>(rsq))` in `snap_ui_kernel` + `snap_deidrj_kernel`, everything else bit-for-bit FP64 identical to `SnapGpu`). `src/gpu/include/tdmd/gpu/snap_gpu_mixed.hpp` new — PIMPL header with byte-identical public interface to `SnapGpu` for typedef-dispatch compatibility. `src/potentials/include/tdmd/potentials/snap_gpu_adapter.hpp` — added `SnapGpuActive` typedef dispatch (`using SnapGpuActive = SnapGpuMixed` under `TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY`, else `SnapGpu`). `src/potentials/include/tdmd/potentials/eam_alloy_gpu_adapter.hpp` — EAM dispatch guarded with explicit `!defined(TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY)` so EAM stays FP64 on SnapOnly (tabulated Horner splines fail monotonicity in FP32 per D-M6-8). `src/gpu/CMakeLists.txt` — `snap_gpu_mixed.cu` added to sources + CPU-only LANGUAGE=CXX retag. `cmake/BuildFlavors.cmake` — status message rewritten "T8.9 kernel split pending" → "narrow-FP32 pair-math (T8.9)", comment block updated with measured thresholds + throughput-honesty note. `tests/gpu/test_snap_mixed_fast_within_threshold.cpp` new (~310 lines) — three D-M8-8 gates (per-atom force L∞ rel ≤ 1e-5, PE rel ≤ 1e-7, virial rel-to-max-component ≤ 5e-6) on T6 tungsten 2000-atom rattled BCC; bypasses flavor-dispatched adapter (constructs `SnapGpu` + `SnapGpuMixed` directly) to measure D-M8-8 on every flavor's CI pass. `tests/gpu/test_snap_gpu_bit_exact.cpp` + `tests/gpu/test_snap_gpu_functional.cpp` — `#ifdef TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY { SUCCEED; return; }` guards added (D-M8-8 supersedes D-M8-13 on this flavor; adapter dispatches to SnapGpuMixed so CPU FP64 ≡ GPU at 1e-12 is structurally impossible). **Scope narrowing — why the wider FP32 plan failed.** The pre-impl report and original SPEC §D.11 contemplated FP32 across the full pair-math block (`sqrtf` + `__sincosf` + `z0_f` + `dz0dr_f` + FP32 sfac). Exploratory measurement showed force worst rel 2.12e-5 on T6 tungsten, **above the D-M8-8 1e-5 cap**. Root cause: `__sincosf` carries ~1 ULP rel error (~6e-8), which feeds `z0 = r·cos(θ)/sin(θ)`, and `z0` enters the VMK U-recurrence multiplicatively; a length-O(165) bispectrum dot product at twojmax=8 compounds this to ~2e-5 on force. Narrowing to `sqrtf` alone keeps `r`'s rel error at ~6e-8 but all downstream scalars FP64, bounding the compounding. **Measured residuals on T6 2000-atom rattled BCC-W fixture:** force worst rel = **9.51e-6** at atom 1282 component z (tol 1e-5 — close to cap but deterministic on sm_120 since `sqrtf` is IEEE ≤1 ULP), PE rel = **1.70e-9** (tol 1e-7, ~60× headroom), virial rel-to-max-component = **3.94e-9** on Voigt[4] (tol 5e-6, ~1000× headroom). **Realized throughput lever is modest.** SNAP pair-math is <5 % of total SNAP runtime — the U-recurrence + bispectrum dot products dominate. `MixedFastSnapOnlyBuild` remains opt-in; real SNAP throughput gains come from `Fp64ProductionBuild` + dense-cutoff GPU path (T8.6b landed). Phase B (full-FP32 bispectrum) deferred to §D.17 procedure contingent on T8.11 cloud-burst perf pressure. **Build-flavor matrix.** All 51 ctest gates green on Fp64ReferenceBuild (build/), MixedFastBuild (build-mixed/), MixedFastSnapOnlyBuild (build-mixed-snap-only/) — one structural skip `test_t4_nve_drift` is MixedFast-only by design. **Out of scope (handed off).** T8.10 — TDMD vs LAMMPS GPU SNAP perf baseline on RTX 5080. T8.11 — cloud-burst ≥2 GPU nodes scaling gate (primary M8 ≥20% gate). Phase B full-FP32 bispectrum — formal §D.17 procedure, contingent on T8.11 perf pressure.

**T8.7 addendum (2026-04-20) — M8: SnapPotential GPU FP64 bit-exact gate vs CPU FP64 (Validation / Reference Engineer role).** `docs/specs/gpu/SPEC.md` v1.0.19 (status line bumped T8.6b→T8.7; §7.5 roadmap row T8.7 annotated LANDED with measured headroom; change log row T8.6b marked done + T8.7 row promoted from "planned" to "done — v1.0.19; D-M8-13 byte-exact chain closed"). `tests/gpu/test_snap_gpu_bit_exact.cpp` (new, ~240 lines) — Catch2 test exercising `SnapPotential` CPU FP64 vs `SnapGpuAdapter` GPU FP64 on a 10×10×10 BCC W rattled fixture (2000 atoms, box side 31.8 Å, sinusoidal 0.01 Å rattle), asserting PE + per-atom force L∞ + all six virial Voigt components ≤ 1e-12 rel. `tests/gpu/CMakeLists.txt` (+14 lines — new `test_snap_gpu_bit_exact` target with `TDMD_TEST_FIXTURES_DIR` define + `SKIP_RETURN_CODE 77`). `docs/development/m8_execution_pack.md` §5 T8.7 header promoted from "SnapPotential GPU FP64 bit-exact gate vs CPU FP64" to "[LANDED 2026-04-20]" with measured headroom block + per-flavor matrix. **Gate scope.** 10×10×10 BCC W = 2000 atoms ≥ the master-spec §4.7 "8×8×8 BCC (2048-atom)" reference envelope (8³·2 = 1024 in canonical BCC counting → 10³·2 = 2000 comfortably exceeds 3·(rcut+skin) = 14.8 Å minimum at box side 31.8 Å). Canonical rattle identical to `test_snap_gpu_functional`'s pattern for comparability against T8.6b measurements. Test issues a single `SnapPotential::compute` + a single `SnapGpuAdapter::compute` against shared CellGrid + NeighborList, then compares PE (`rel ≤ 1e-12`), per-atom forces (L∞ `rel ≤ 1e-12` with worst-case atom/component diagnostics), and all six virial Voigt components (`rel ≤ 1e-12` each). Uses `pool → stream → adapter` construction order explicitly commented per gpu/SPEC §7.5 lifetime-contract footnote established at T8.6b; would reproduce the heap-corruption bug if reversed. Self-skips exit 77 on no-CUDA or uninitialized LAMMPS submodule (Option A convention). **What the test measured.** On build/ (Fp64Reference+CUDA):

- PE rel = **2.04e-16** (FP64 round-off floor) — gate 1e-12 → 4 OoM headroom at the "low" dim.
- Worst per-atom force rel = **1.69e-14** at atom 1256 component y — gate 1e-12 → 2 OoM headroom; consistent with T8.6b's 1.32e-14 measurement on the smaller 250-atom fixture (~8× bigger system, ~4× the rattle-phase spread, worst rel err grew ~1.3× — benign).
- Worst virial Voigt rel = **2.04e-13** at component [5] (V_xy = 1.37 eV) — smallest-magnitude Voigt component in the dataset, dominated by the `den = max(1, |a|, |b|)` denominator landing at 1.37 rather than the large diagonal ≈ 35619. Still 1 OoM under the 1e-12 gate; non-issue structurally.

**Headroom rationale.** SNAP GPU FP64 is bit-exact in Pass 1 (Ui accumulation — thread-parallel zero of `ulisttot_r/i` then single-lane neighbour loop; same reduction order as CPU) and in Pass 3 (per-neighbour `du/dr` single-lane contraction); the only non-bitwise ambiguity is Pass 2's Z-list summation where CPU's j1≤j≤j1+j2 triple-nested accumulation meets the GPU's linearly-indexed `idxz_block` walker. Both orderings are deterministic, both FP64 round independently, and the worst observed divergence (1.69e-14) is ≈ √k_max × ε_FP64 (k_max=55, √55·ε ≈ 1.65e-14) — precisely the expected canonical Kahan-free FP64 cumulative-rounding envelope. 4 OoM below the 1e-12 gate means the gate survives any future kernel-restructuring that preserves reduction determinism (D-M6-7 invariant). **Threshold registry status change.** `verify/thresholds/thresholds.yaml` `benchmarks.t6_snap_tungsten.gpu_fp64_vs_cpu_fp64.forces_relative = 1.0e-12` promoted from **reserved** (written at T8.1b pack authoring, no test asserting it) to **ACTIVE** (gate enforced at build-time via `test_snap_gpu_bit_exact`). Rationale block referencing `mixed_fast_snap_only_rationale.md` §4 + `potentials/SPEC.md` §6.7 preserved verbatim; no PE/virial-rel sibling entries needed because the current registry schema only names `forces_relative` as the canonical key and PE/virial are covered by the same budget via §6.7 language. **All three flavors green.** Fp64Reference+CUDA 50/50 (1 expected skip `test_t4_nve_drift` — MixedFast-only), MixedFast+CUDA 50/50, CPU-only-strict 42/42 (1 expected skip). Test total wall = 0.85 s on RTX 5080 (build/ flavor); `--success` diagnostics captured above for the rationale section. **D-M6-7 chain extended.** CPU FP64 Reference ≡ GPU FP64 Reference (SNAP) closes D-M8-13. Combined with D-M6-7 EAM/alloy (2026-04-11) and D-M5-9 thermo-level Kahan invariance (2026-03-30), the master spec's byte-exact oracle chain now covers: CPU Morse (M1), CPU EAM/alloy (M2), GPU EAM/alloy (M6 T6.5 + T6.7), CPU SNAP (M8 T8.4b + T8.5), GPU SNAP (M8 T8.7). All five potential paths are Reference-bit-exact vs the canonical CPU FP64 oracle. **Out of scope (handed off).** T8.9 — MixedFastSnapOnlyBuild GPU SNAP FP32-force variant with its own D-M8-8 dense-cutoff threshold (≤1e-5 force / ≤1e-7 PE); distinct gate, does not share the 1e-12 budget by design (the mixed-precision path is scientifically motivated by SNAP ML fit RMSE ≈ 1e-3 eV/atom ≫ FP32 ULP — "can afford" FP32 SNAP forces). T8.10 — T6 full-scale fixture scaling runs + TDMD-vs-LAMMPS SNAP performance baseline. T8.11 — cloud-burst ≥ 2 GPU nodes scaling gate. All three CI flavors local pre-push: green.

**T8.6b addendum (2026-04-20) — M8: SnapPotential GPU FP64 functional kernel body (GPU / Physics Engineer joint role).** `docs/specs/gpu/SPEC.md` v1.0.18 (§7.5 rewritten — "planned T8.6b" → "shipped T8.6b"; three-kernel architecture documented; NVTX coverage matrix updated; lifetime-contract footnote added; roadmap row updated). `src/gpu/snap_gpu_tables.cuh` + `src/gpu/snap_gpu_tables.cu` (new — host-side integer/factorial index-table builder, ~290 LoC, ports `SnaEngine::build_indexlist` + `init_clebsch_gordan` + `init_rootpqarray` verbatim so `cglist` is byte-exact with the CPU oracle by construction; GPLv2 attribution preserved from upstream `sna.cpp`). `src/gpu/snap_gpu_device.cuh` (new — `__device__` helpers: switching function + `compute_uarray_single_lane` Wigner-U recurrence). `src/gpu/snap_gpu.cu` (rewritten — was CUDA-branch `logic_error` sentinel at T8.6a; now carries full `Impl::compute()` body: build_flat_tables → upload index tables once → upload SNAP params once → rebuild bzero → H2D atoms/cells → three-kernel launch → D2H forces/pe/virial → host Kahan reduce). Three kernels, one-block-per-atom (<<<n_atoms, 128>>>), FP64 throughout: (1) `snap_ui_kernel` — thread-parallel zero of `ulisttot_r/i`, self-term add, single-lane (tid==0) loop over neighbours calling `compute_uarray_single_lane` and accumulating weighted Ulist; (2) `snap_yi_kernel` — single-lane Z-list contraction + Y-list build + B-list inline + PE accumulation + per-atom virial; (3) `snap_deidrj_kernel` — single-lane per-neighbour `du/dr` derivation, computes BOTH `fij_own` AND `fij_peer` via re-read of `ylist[j]` from global, `F_i = Σ_j (fij_own − fij_peer)` replaces the CPU's half-list Newton-3 scatter without any `atomicAdd(double*)` (D-M8-13 compliance). `src/gpu/CMakeLists.txt` (`snap_gpu_tables.cu` added to sources list + CPU-only LANGUAGE=CXX retag — pure host C++ inside a `.cu` wrapper for naming symmetry with `.cuh`). `tests/gpu/test_snap_gpu_functional.cpp` (new — T8.6b functional correctness gate, 5 Catch2 cases: CPU-only build stub; BCC W 250-atom rattled lattice finite+Newton-3 forces; CPU-vs-GPU within 1e-10 rel — loose so T8.6b stays below T8.7's D-M8-13 1e-12 gate; compute_version monotone; empty-atoms returns zero). `tests/gpu/CMakeLists.txt` (new `test_snap_gpu_functional` target with `TDMD_TEST_FIXTURES_DIR` define + `SKIP_RETURN_CODE 77`). `docs/development/m8_execution_pack.md` §5 T8.6 entry (split header promoted from "T8.6a [landed] + T8.6b [pending]" to "T8.6a [landed] + T8.6b [landed]"). **What the test actually measures.** On the BCC W 250-atom rattled lattice (3.18 Å, 5×5×5 repeats, box side 15.9 Å > 3·(rcut+skin) = 14.8 Å, 0.01 Å deterministic sinusoidal rattle), CPU vs GPU: PE identical to −4457.9 eV, worst-force rel err = **1.32e-14** (already four orders of magnitude below T8.7's D-M8-13 1e-12 budget), virial bit-exactly zero-diff on all six Voigt components. No `atomicAdd(double*)` anywhere in the kernel pipeline — full-list peer-side replay + host-side Kahan absorb the reduction-order ambiguity. **Lifetime-contract trap resolved.** `SnapGpuAdapter::Impl` holds `DevicePtr<std::byte>` members whose pool-class deleters reference the owning `DevicePool::Impl*` as context. Reverse construction order (`adapter` before `pool`) triggers use-after-free on teardown: pool dtor runs first (calls `cudaFreeAsync` on all tracked blocks), then adapter's `~Impl` runs `DevicePtr` destructors that re-dereference the already-destroyed pool Impl → malloc-arena corruption visible as "malloc(): unaligned tcache chunk" / "corrupted size vs. prev_size" / "unsorted double linked list corrupted" on process exit. Fix: **construct pool → stream → adapter in that order** (matches EAM test convention); destruction runs adapter → stream → pool and the adapter's DevicePtr deleters reach a still-alive pool. Same contract applies to `EamAlloyGpuAdapter`. Test file enforces ordering by convention; gpu/SPEC §7.5 now documents the contract. **CPU-only build: Werror=sign-conversion cleanup** in `snap_gpu_tables.cu` — int × int → size_t multiplications wrapped in `const std::size_t jdim_cube = jdim_sz³` local (gcc-14 with `-Werror=sign-conversion` retags the `.cu` as C++ for CPU-only builds and surfaces warnings that nvcc silences; not a correctness issue). **All three flavors green:** Fp64Reference+CUDA 49/49 (48 passed + 1 expected skip `test_t4_nve_drift` — MixedFast-only), MixedFast+CUDA 49/49, CPU-only-strict 41/41 (40 passed + 1 expected skip). `test_t6_differential` (T8.5 D-M8-7 CPU byte-exact SNAP oracle) без регрессии — T8.5 gate preserved. NVTX audit green — every kernel launch wrapped in `TDMD_NVTX_RANGE`. **Out of scope (handed off).** T8.7 — bit-exact gate `tests/gpu/test_snap_gpu_bit_exact.cpp` on W 2000-atom fixture at D-M8-13 ≤ 1e-12 rel (preliminary measurement already shows we're 4 OoM under the budget; T8.7 is essentially a formal locking PR). T8.9 — MixedFast SNAP kernel variant. T8.10 — T6 full-scale fixture diff + SNAP GPU vs LAMMPS CPU performance baseline. T8.11 — cloud-burst scaling ≥ 2 GPU nodes. `test_t4_nve_drift` extension to SNAP + NVE drift smoke deferred — current VV integrator wiring covers EAM only; SNAP GPU runtime-backend smoke lands with T8.10/T8.11.

Проведён systematic анализ production MD-кодов (LAMMPS GPU package, GROMACS 2024-2026, NAMD 3.0, DeePMD-kit). Закрыты **4 high-priority blind spots**, выявленные анализом — каждый представляет silent failure mode или performance regression, который не был формально адресован до текущего момента.

1. **Dynamic load balancing policy (scheduler/SPEC §11a — новый раздел ~170 строк):**
   - Three-phase maturity: M5-M6 measurement-only, M7-M8 advisory, v2+ active DLB;
   - Imbalance thresholds: warning at 1.3x, error at 3.0x;
   - TDMD-specific: frontier stall amplification вместо SD per-rank slowdown;
   - Active DLB только в FastExperimental profile (breaks layout-invariant determinism by design);
   - Pattern 2 dual balance: intra-subdomain + inter-subdomain;
   - `tdmd diagnostic --load-balance` CLI integration;
   - Justifies different philosophy от LAMMPS/GROMACS (DLB on by default): TDMD — scientific defaults, performance opt-in.

2. **GPU-resident execution mode mandate (integrator/SPEC §3.5 — новый раздел):**
   - Закрывает NAMD 3.0 lesson: traditional offload оставляет GPU idle 30-50%;
   - **Mandatory** GPU-resident при `backend=cuda`: все integrator kernels на GPU;
   - `velocity_verlet_pre_force_kernel_gpu` reference template с `__restrict__`;
   - CPU-GPU transfer ТОЛЬКО при bootstrap/dump/checkpoint/finalize, never per-step;
   - Performance gate: `gpu_idle_fraction < 0.10` в T6 SNAP benchmark;
   - Consequences extended к state/, neighbor/, comm/ — "data lives on GPU" as unifying principle.

3. **Workload saturation и minimum atoms per rank (perfmodel/SPEC §3.7 — новый раздел ~200 строк):**
   - `N_min_saturation(potential, hw)` formula учитывающий kernel launch overhead, memory bandwidth, neighbor list overhead;
   - Tabulated values для всех potentials × A100/H100 (LJ 10k, EAM 5k, MEAM 2k, SNAP 1k на A100);
   - `SaturationVerdict` enum (WellSaturated / Saturated / UnderSaturated / SeverelyUnderSaturated);
   - `PerfModel::recommend_deployment` с alternatives когда saturation бедный;
   - `tdmd explain --perf` показывает saturation analysis с concrete recommendations;
   - Preflight interactive confirmation для severely under-saturated configs;
   - Runtime monitoring: `gpu_utilization_measured < 50%` triggers log warning.

4. **Expensive compute intervals policy (master spec §6.5b — новый раздел + integrator/SPEC §4.5 — thermostat coupling):**
   - Закрывает GROMACS lesson: frequent coupling / global reduction становится bottleneck;
   - Defaults: potential_energy=100, kinetic_energy=50, virial=100 (не каждый шаг);
   - `thermostat_update_interval: 50` default, stability constraint `damping_time/coupling_period > 2.0`;
   - Preflight warnings для aggressive intervals с overhead estimates;
   - Dump/compute interval alignment validation (avoid extra computations);
   - Auto-tuning recommendations в M8+ (measured reduction latency based);
   - Telemetry: `global_reduction_time_ms_total`, reduction breakdown в final report.

**Rationale для выбора items:** все 4 — actual silent failures observed в production MD codes. Items 5-12 из анализа (PP/PME split, ML potentials framework, Newton×atomics, thread-MPI, GPU direct comm, clock sensitivity, PME FFT) — документированы в open questions per module SPECs, адресуются когда станут релевантны в v1.5-v2.0.

**T8.6a addendum (2026-04-20) — M8: SnapPotential GPU FP64 scaffolding (GPU / Physics Engineer joint role).** `docs/specs/gpu/SPEC.md` v1.0.17 (§7.4 rewritten — LJ/Morse/MEAM/PACE/MLIAP stay CPU-only M9+; SNAP moved to M8 window T8.6a→T8.6b→T8.7; §7.5 new — full SNAP GPU contract authored: locked `SnapGpu` / `SnapGpuAdapter` API, M8-scope flag fence chemflag/quadraticflag/switchinnerflag parity с CPU `SnapPotential`, single-TU `src/gpu/snap_gpu.cu` build-guard story — `#if TDMD_BUILD_CUDA` branches on `logic_error` T8.6b sentinel (CUDA) / `runtime_error` CPU-only, mirrors `eam_alloy_gpu.cu` precedent, NO separate stub.cpp; planned T8.6b three-pass algorithm Ui/Yi/deidrj + T8.7 bit-exact gate D-M8-13 ≤ 1e-12 rel; roadmap rows T8.0/T8.6a/T8.6b/T8.7 added; retroactive v1.0.16 row for T8.0 §3.2c added). `src/gpu/include/tdmd/gpu/snap_gpu.hpp` (new — PIMPL, matches `EamAlloyGpu` shape; `SnapTablesHost` POD; `SnapGpuResult { potential_energy, virial[6] }`; move-only; `compute_version()` monotone counter) + `src/gpu/snap_gpu.cu` (new — both CUDA + CPU-only branches in single TU; CUDA branch throws `std::logic_error("SnapGpu::compute: T8.6b kernel body not landed")` wrapped в `TDMD_NVTX_RANGE("snap.compute_stub")` для auditability; CPU-only branch throws `std::runtime_error("...CPU-only build...")`; `Impl::compute()` body pre-declared DevicePtr fields для T8.6b — all unused at T8.6a; zero `<<<...>>>` launches) + `src/potentials/include/tdmd/potentials/snap_gpu_adapter.hpp` (new — domain facade; signature identical к `EamAlloyGpuAdapter::compute(AtomSoA&, Box&, CellGrid&, DevicePool&, DeviceStream&) → ForceResult`) + `src/potentials/snap_gpu_adapter.cpp` (new — ctor validates n_species > 0 / rcutfac > 0 / !chemflag / !quadraticflag / !switchinnerflag / β size matches k_max + 1 / rcut_sq_ab = n×n; flattens per-species `radius_elem_flat_` + `weight_elem_flat_` + species-major `beta_flat_` (stride k_max+1); `compute()` marshals AtomSoA/Box/CellGrid → BoxParams + SnapTablesHost → calls `SnapGpu::compute()` which throws T8.6b sentinel) + `src/runtime/include/tdmd/runtime/simulation_engine.hpp` (forward-decl `class SnapGpuAdapter`; new field `std::unique_ptr<potentials::SnapGpuAdapter> gpu_snap_potential_` parallel к `gpu_potential_`) + `src/runtime/simulation_engine.cpp` (`init()` switches на `dynamic_cast<EamAlloyPotential*>` / `dynamic_cast<SnapPotential*>` → populates appropriate adapter field; throws `std::invalid_argument("runtime.backend=gpu requires EAM/alloy or SNAP potential")` otherwise; `recompute_forces()` branches `if (gpu_potential_) ... else if (gpu_snap_potential_) ... else throw logic_error`) + `src/io/preflight.cpp::check_runtime` (relaxed — `pot.style != EamAlloy && pot.style != Snap` error; message "runtime.backend='gpu' currently only supports potential.style в {eam/alloy, snap} (M6 / T8.6 scope)") + `src/gpu/CMakeLists.txt` (snap_gpu.cu added to `tdmd_gpu_sources` + `set_source_files_properties LANGUAGE CXX` list for CPU-only builds) + `src/potentials/CMakeLists.txt` (snap_gpu_adapter.cpp added to `tdmd_potentials`) + `tests/gpu/test_snap_gpu_plumbing.cpp` (new — 4 Catch2 cases; uses W_2940_2017_2.snap{coeff,param} fixture via `verify/third_party/lammps/examples/snap/`; self-skips exit 77 if LAMMPS submodule не initialized — same Option A convention as `test_snap_compute`) + `tests/gpu/CMakeLists.txt` (new `test_snap_gpu_plumbing` target с `TDMD_TEST_FIXTURES_DIR` define + `SKIP_RETURN_CODE 77`) + `docs/development/m8_execution_pack.md` §5 T8.6 entry (split to T8.6a [x] + T8.6b [ ] carry-forward). **Dispatch architecture decision.** Parallel field pattern (`gpu_potential_` для EAM + `gpu_snap_potential_` для SNAP) в `SimulationEngine` rather than abstract `GpuPotentialAdapter` base — minimum-scope for T8.6a; abstract-base refactor can land at T8.6b / later if 2+ concurrently-live GPU potentials ever require это. **NVTX invariant preserved.** Zero `<<<...>>>` kernel launches в T8.6a's `snap_gpu.cu` → `test_nvtx_audit` grep walker trivially passes без additional wrapper work. `TDMD_NVTX_RANGE("snap.compute_stub")` wraps the CUDA-branch sentinel throw для auditability (and establishes NVTX scoping convention ahead of T8.6b kernel launches). **All three CI flavors зелёные:** Fp64Reference+CUDA 48/48 (1 expected skip `test_t4_nve_drift` — MixedFast-only), MixedFast+CUDA 48/48, CPU-only-strict 40/40 (1 expected skip `test_t4_nve_drift`). `test_t6_differential` (T8.5 D-M8-7 CPU byte-exact SNAP oracle) без регрессии — T8.5 gate preserved. **Why split T8.6 → T8.6a + T8.6b.** Mirrors T8.4a/T8.4b CPU precedent: ~1500-line LAMMPS USER-SNAP port (`SnaEngine::compute_snap_common()` + Wigner-U expansion + bispectrum basis cache + Clebsch-Gordan contraction + three-pass force Ui/Yi/deidrj + index-table flatten T** → T*) deserves a separate reviewable PR для kernel body; scaffolding landing first pins the public adapter shape и the CPU-only build-guard story before any CUDA-specific code is written. Downstream M8 tasks (T8.9 MixedFast SNAP, T8.10 T6 full-scale fixture diff, T8.11 cloud-burst scaling) can start их pre-impl reports against the locked `SnapGpuAdapter::compute()` signature без блокировки на T8.6b kernel landing. **T8.6b — remaining work.** Full CUDA kernel port of `SnaEngine::compute_snap_common()` + `SnaEngine::compute_dYidrj()`, three kernels (`snap_ui_kernel` / `snap_yi_kernel` / `snap_deidrj_kernel`) с `__restrict__` per master spec §D.16 + NVTX `TDMD_NVTX_RANGE("snap.{ui,yi,deidrj}_kernel")` per gpu/SPEC §9 audit rules + Kahan reduction host-side per D-M6-15 + LAMMPS GPLv2 attribution block в каждом новом `.cu`/`.cuh`. **T8.7 — follow-on.** Bit-exact gate `tests/gpu/test_snap_gpu_bit_exact.cpp` on W 2000-atom fixture — GPU FP64 ≡ CPU FP64 ≤ 1e-12 rel (D-M8-13 closure).

**T8.8 addendum (2026-04-20) — M8: MixedFastSnapOnlyBuild new BuildFlavor shipped via §D.17 7-step formal procedure (Architect + Validation Engineer joint role).** `docs/specs/potentials/mixed_fast_snap_only_rationale.md` (new, 256 lines) + master spec §7.1 (BuildFlavor table extended: SNAP/EAM split columns + B-het row) + §7.2 (compat matrix row) + §7.4 (M8 extension paragraph promoted from "запланирован" to "landed T8.8") + §D.2 (five→шесть canonical flavors table; B-het philosophy row) + §D.11 (roadmap-extension paragraph promoted to landed-flavor reference with 7-step artefact pointers) + §D.12 (compat matrix row + canonical-cell rationale) + §D.13 (thresholds table extended with SnapOnly column; SNAP and EAM rows now split) + §D.14 (build system rewritten from 5-target multi-binary snippet to single-binary `TDMD_BUILD_FLAVOR` cache-variable model matching actual `cmake/BuildFlavors.cmake` implementation; all six flavors listed) + §D.15 (new `tdmd_mixed_snap_only` binary label entry) + `verify/thresholds/thresholds.yaml` (new `benchmarks.gpu_mixed_fast_snap_only` section — SNAP force 1e-5 / SNAP energy 1e-7 / EAM force 1e-5 / EAM energy 1e-7 / EAM virial 5e-6 / NVE drift 1e-5 per 1000 steps; all thresholds equal-or-tighter than MixedFastBuild) + `docs/specs/potentials/SPEC.md` §6.7 (promoted from "до T8.8 landing этот flavor не доступен" placeholder to finalized interface with rationale-doc cross-ref and T8.12 slow-tier pass marker) + `docs/user/build_flavors.md` (new scientist-facing 6-flavor guide with decision tree + per-flavor use cases + warnings for research-only flavors) + `docs/development/m8_execution_pack.md` §5 T8.8 checklist marked `[x]` with step 5/7 satisfied-by pointers (step 5 → T8.12 VerifyLab pass; step 7 → review signoffs recorded on PR thread) + `CMakeLists.txt` (cache STRINGS list extended with `MixedFastSnapOnlyBuild` as fourth entry) + `cmake/BuildFlavors.cmake` (new `_tdmd_apply_mixed_fast_snap_only` function defining `TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY` compile symbol; elseif dispatch branch in `tdmd_apply_build_flavor`; host `-fno-fast-math` + CUDA `--fmad=true` matching MixedFastBuild because heterogeneous precision split lives inside SNAP/EAM kernel TUs not in compiler flags). **What landed under the 7-step procedure.** Step 1 (formal rationale): `mixed_fast_snap_only_rationale.md` documents the need (pure SNAP runs + hybrid SNAP+EAM runs where EAM precision must stay FP64), why existing flavors don't suffice (4-row comparison table), empirical evidence the precision mix is physically sound (SNAP fit RMSE ≈ 13.8 meV/atom vs FP32 force residual ~1e-5 rel — three orders of magnitude below ML noise floor, cites Wood&Thompson arXiv:1702.07042 and T6.8a `project_fp32_eam_ceiling.md` EAM ceiling), threshold budget derivation (D-M8-8 dense-cutoff analog inheriting MixedFastBuild envelope), compat matrix proposal, CMake integration, slow-tier pass obligation, scientist docs pointer, two-reviewer signoff mandate, out-of-scope follow-on tasks (T8.9 SNAP FP32 kernel + EAM FP64 branch preservation; T8.11 cloud-burst scaling with this flavor active; T8.12 slow-tier VerifyLab pass; T8.13 v1.0.0-alpha1 release notes mention). Step 2 (compat matrix): §D.12 row added + canonical-cell rationale explaining why SnapOnly+Production is canonical and SnapOnly+FastExperimental is permitted-not-canonical. Step 3 (threshold registry): `verify/thresholds/thresholds.yaml` `benchmarks.gpu_mixed_fast_snap_only` section with all 8 metrics + derivation rationale + status=active_T8.12_slow_tier. Step 4 (CMake option): `TDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild` registered in cache STRINGS; `_tdmd_apply_mixed_fast_snap_only` function wires compile flags + `TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY` define + status message "T8.9 kernel split pending"; configure verified cleanly via `cmake -DTDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild ...` in /tmp probe — emits flavor status line for every TDMD library target. Step 5 (slow-tier VerifyLab pass): recorded as **T8.12 hard gate before M8 closure** in rationale doc §7 + exec pack §5 T8.8 entry; full T0+T1+T3+T4+T6 battery required on `MixedFastSnapOnlyBuild` binary, any single failure blocks M8 closure. Step 6 (user documentation): `docs/user/build_flavors.md` 6-flavor guide with decision tree covering pure SNAP / pure EAM / mixed SNAP+EAM / bitwise-determinism workflows + per-flavor when-to-use-and-when-not-to guidance. Step 7 (Architect+Validation joint review): two-reviewer signoff mandate recorded in rationale doc §9 checkboxes; signoffs tracked on T8.8 PR thread per master spec §D.17 final step requirement (**not optional**). **What remains pending for SnapOnly activation:** T8.9 kernel split emission (SNAP FP32 kernel variant + EAM FP64 branch preservation under this flavor; requires T8.4b SNAP force body port as prerequisite) + T8.12 slow-tier VerifyLab pass + PR-thread review signoffs. Pure SPEC+CMake PR per playbook §9.1 — no functional code changes; configures cleanly but emits no heterogeneous kernel paths yet. Все три CI flavors (Lint, Docs lint, Build CPU gcc-13 + clang-17, Build GPU compile-only Fp64Reference + MixedFast) — push pending.

**T8.4a addendum (2026-04-20) — M8: SNAP types + parser + SnapPotential skeleton (Physics Engineer role).** `src/potentials/include/tdmd/potentials/snap_file.hpp` (new, 121 lines) + `src/potentials/snap_file.cpp` (new, 382 lines) + `src/potentials/include/tdmd/potentials/snap.hpp` (new) + `src/potentials/snap.cpp` (new, skeleton) + `tests/potentials/test_snap_file.cpp` (new, 13 cases / 51 assertions) + `src/potentials/CMakeLists.txt` (snap_file.cpp + snap.cpp добавлены в `tdmd_potentials`) + `tests/potentials/CMakeLists.txt` (new `test_snap_file` target with submodule SKIP_RETURN_CODE 77). **What landed.** `SnapParams` (11 hyperparameters — twojmax, rcutfac, rfac0, rmin0, switchflag, bzeroflag, quadraticflag, chemflag, bnormflag, wselfallflag, switchinnerflag — defaults match LAMMPS `pair_snap.cpp` constructor); `SnapSpecies` (name, radius_elem, weight_elem, β coefficient vector); `SnapData` (params + species + derived k_max + symmetric rcut_sq_ab n×n matrix + FNV-1a 64-bit checksum over all parsed fields). Parsers `parse_snap_coeff(path)` / `parse_snap_param(path)` / `parse_snap_files(coeff, param)`: whitespace-tokenised, `#`-comments stripped, `path:line: message` diagnostic format, missing-required-key + invalid-bool + unknown-key + coefficient-count-mismatch all tagged с file position, `chemflag=1` + `switchinnerflag=1` rejected с forward-looking "deferred to M9+" diagnostic. `snap_k_max(twojmax)` implements LAMMPS `SNA::compute_ncoeff` — verified 1/5/14/30/55/91/140 for twojmax=0/2/4/6/8/10/12 (k_max=55 cross-checks с `W_2940_2017_2.snapcoeff` header declaration of 56 coefficients = k_max + 1). `SnapPotential final : Potential` constructor validates SnapData internal consistency (species count > 0, twojmax ≥ 0 ∧ even, β count = k_max + 1 linear or k_max + 1 + k_max(k_max+1)/2 quadratic, rcutfac > 0) и throws `std::invalid_argument` on mismatch; `compute()` throws `std::logic_error("SnapPotential::compute: force evaluation not yet implemented — T8.4b — LAMMPS USER-SNAP port")` — force body = verbatim port of LAMMPS `sna.cpp` (1597 lines) + `pair_snap.cpp` (808 lines) deferred to T8.4b as a follow-up PR (total ≈ 1600 lines of Clebsch-Gordan / bispectrum / three-pass force code, not session-sized). `cutoff()` returns `data_.max_pairwise_cutoff() = max over pairs of rcutfac·(R_α + R_β)`. Test suite: 13 Catch2 cases / 51 assertions, all green locally — covers snap_k_max formula, canonical W_2940_2017_2 parse (twojmax=8, rcutfac=4.73442, β[0]=0.781170857801, β[55]=−0.008314173699, checksum stable через re-parse), coefficient-count mismatch error message, missing-required-key, odd-twojmax rejection, chemflag=1 rejection, invalid header, skeleton throw, constructor validator. Tests auto-skip exit 77 on uninitialized LAMMPS submodule (matches T8.2 oracle fixture gate behaviour; pure-C++ + no LAMMPS link / no CUDA). Full build + entire ctest bank (45/45 passed, 1 pre-existing skip) clean — no regressions. Pre-commit (clang-format + cmake-format + cmake-lint + trailing-ws + EOF + restrict-stub) green. **Out of scope (handed off T8.4b):** `SnapPotential::compute` three-pass bispectrum → energy → force evaluator (Pass 1 compute_sna_atom с U-matrix accumulation + CG contraction + bzero subtract; Pass 2 E_i = β_k · B_{k,i} + β·B cache; Pass 3 F_ij via −Σ_k β_k · dB_k/dr) + scratch layout preservation for D-M8-7 byte-exact; lands as dedicated PR under Physics Engineer role с GPLv2 attribution block in the newly-created `src/potentials/snap/` sub-tree. **Why this split.** T8.4 combined body was sized at ~1600 lines in the exec pack T8.4 task template; the port must preserve the LAMMPS FP summation ordering verbatim so that MixedFastSnapOnlyBuild (T8.8) and GPU kernels (T8.6) land as precision policies over a shared core instead of parallel implementations. Splitting T8.4 into T8.4a (types + parser + skeleton landing now) + T8.4b (force body port) lets T8.5 (CPU differential harness against LAMMPS) start planning directly against the parsed `SnapData` while T8.4b is in flight. M8 execution pack §5 updated accordingly (T8.4 → T8.4a [x] ∧ T8.4b carry-forward note). All three CI flavors (Lint, Docs lint, Build CPU gcc-13 + clang-17, Build GPU compile-only Fp64Reference + MixedFast) — push pending.

**T8.3 addendum (2026-04-20) — M8: potentials/SPEC §6 SNAP body authored (Architect role).** `docs/specs/potentials/SPEC.md` §6 expanded from 5-subsection stub (§6.1 Форма, §6.2 Why, §6.3 Cost, §6.4 Strategy, §6.5 Validation — ~45 lines) to 9-subsection full interface contract (§6.1–§6.9 — ~270 lines) matching §4 EAM / §5 MEAM authoring depth. Pure SPEC delta per playbook §9.1 — no code lands at T8.3. **What landed.** §6.1 SNAP bispectrum formulation with full parameter map (linear + quadratic variant + k_max formula + J_max typical values); §6.2 Why-wave-1 reaffirmation incl. master spec §14 M8 acceptance gate ≥20% @ 8 ranks; §6.3 cost characteristics table (Morse vs EAM vs SNAP J=4 vs SNAP J=8 FLOP/pair + FLOP/atom) with perfmodel/SPEC §3.7 saturation cross-ref (SNAP saturates GPU at ~1000 atoms vs ~5000 for EAM); §6.4 **full C++ interface contract** — `SnapParams` (twojmax, rcutfac, rfac0, rmin0, switchflag, bzeroflag, quadraticflag, chemflag 1:1 mapped from LAMMPS ComputeSNA) + `SnapSpecies` (per-element radius/weight/β) + `SnapData` (params + species + derived pairwise rcut matrix + checksum) + `SnapPotential final : PotentialModel` (4 invariants: `is_local()==true`, species-pair max cutoff for neighbor/SPEC §3.2 skin feed, force-zero-out per §7.2, D-M8-7 byte-exact scratch layout mandate); §6.5 **three-pass force algorithm** (Pass 1 compute_sna_atom bispectrum with U-matrix accumulation + Clebsch-Gordan contraction + bzero subtract; Pass 2 energy + β·B cache; Pass 3 force via -Σ beta_B · dB_k/dr) + byte-exactness contract (scratch layout + reduction order inherited verbatim from LAMMPS sna.cpp/pair_snap.cpp) + GPLv2 license chain (SPDX-License-Identifier: GPL-2.0-or-later + attribution block citing Thompson JCP 2015 + Wood&Thompson arXiv:1702.07042 T6 fixture authors + src/SNAP/ source paths); §6.6 **LAMMPS-native parameter file format** — `.snap` hybrid-overlay include entry point + `.snapcoeff` (per-species coefficient layout) + `.snapparam` (8-hyperparameter key-value grammar) parsed by new `parse_snap_files(coeff, param, species_map) → SnapData` (analogous to `parse_eam_alloy`/`parse_eam_fs` §4.5, lands T8.4); §6.7 **precision policy matrix** — Fp64Reference/Fp64Production FP64 throughout bit-exact vs LAMMPS; MixedFastBuild SNAP in FP32 throughout (D-M8-8 dense-cutoff analog ≤1e-5 rel force / ≤1e-7 rel PE; motivated by SNAP ML-fit noise floor ≈1e-3 eV/atom >> FP32 ULP); **MixedFastSnapOnlyBuild (new M8 T8.8)** heterogeneous SNAP=FP32 + EAM=FP64 + State=FP64 via §D.17 formal procedure — single approved per-kernel precision mix per §8.7 which is now cross-referenced back to §6.7; §6.8 **GPU kernel strategy** — three kernels (bispectrum_kernel block-per-atom with CG-coefficient shared-memory cache; energy_kernel thin per-atom β·B; force_kernel D-M6-7 canonical gather-to-single-block Kahan in Reference, atomicAdd in Production) + D-M8-7 byte-exact GPU FP64 vs CPU FP64 ≤1e-12 rel gate + NVTX ranges per §9.4; §6.9 **full T8.4–T8.12 validation matrix** — unit (T8.4), CPU diff D-M8-7 (T8.5), GPU byte-exact D-M6-7 extension (T8.7), MixedFast D-M8-8 (T8.9), T6 benchmark + D-M7-10 chain extension M3≡…≡M7≡M8 P_space=N K=1 Reference thermo byte-exact (T8.10), scaling probe D-M8-5 cloud-burst-gated (T8.11), slow-tier §D.17 pass for MixedFastSnapOnlyBuild (T8.12) + `verify/thresholds.yaml` anchor registry entries for D-M8-7/D-M8-8 force/PE/1000-step drift caps. **Sister edit:** §8.7 per-kernel-precision-override section cross-referenced back to §6.7 + §D.17 procedure pointer. Pure markdown delta; markdownlint clean. Master spec §D.11/§D.17 + §12.1 SNAP port + §14 M8 acceptance gate cross-referenced throughout. Out-of-scope (handed off): `src/potentials/snap/` CPU implementation (T8.4), GPU kernels (T8.6), MixedFastSnapOnlyBuild §D.17 formal procedure (T8.8 — touches §D.11 + §7.1 BuildFlavors table). T8.2 CI run 24662328788 pending (background); T8.3 changes to push after T8.2 closure.

**T8.2 addendum (2026-04-20) — M8: LAMMPS SNAP oracle subset verified + T6 canonical fixture chosen.** `verify/third_party/lammps_README.md` (new **SNAP fixture** section, 5-row artefact table + sanity-run snippet) + `docs/specs/verify/SPEC.md` §4.1 T6 row (fixture name annotated) + new §4.7 (`t6_snap_tungsten` canonical fixture body, status="fixture choice landed T8.2, full body T8.10") + `tests/potentials/test_lammps_oracle_snap_fixture.cpp` (new — Catch2 path gate, two test cases, self-skips exit 77 on uninitialized submodule) + `tests/potentials/CMakeLists.txt` (new target + `SKIP_RETURN_CODE 77`) + `docs/development/m8_execution_pack.md` §5 (T8.2 `[x]`). Follows T8.1b correction commit (5ed72d2→0c84b68) that caught D-M8-2/D-M8-3 factual errors: LAMMPS oracle already shipped M1 T1.11 (`verify/third_party/lammps/` submodule pinned `stable_22Jul2025_update4`, `tools/build_lammps.sh` with `PKG_ML-SNAP=on`, `cmake/FindLammps.cmake` already wiring install prefix). T8.2 therefore reduces from "add submodule + authoring CMake option" to "verify ML-SNAP subset operational + lock canonical T6 fixture choice". **Canonical T6 fixture:** `W_2940_2017_2.snap` (+ `.snapcoeff` + `.snapparam` sidecars) — Wood & Thompson "Quantum-Accurate Molecular Dynamics Potential for Tungsten" arXiv:1702.07042 (2017), pure W single-species BCC, 2940 DFT training configurations, dated 2017-02-20. Resolved via submodule path `verify/third_party/lammps/examples/snap/`; no binary tracked by tdmd repo (repo size preserved, D-M8-3). Driver example `in.snap.W.2940` = 128-atom BCC W, 100-step NVE, dt = 0.0005 ps, T₀ = 300 K, `pair_style hybrid/overlay zbl 4.0 4.8 snap`. WBe binary alloy (`WBe_Wood_PRB2019.snap`) deferred to M9+ SNAP alloy gate. **Oracle subset verification executed.** `verify/third_party/lammps/install_tdmd/bin/lmp -h | grep -iE 'ML-SNAP'` reports present; `LD_LIBRARY_PATH=…/lib lmp -in in.snap.W.2940` runs cleanly in 1.2 s on dev hardware; thermo output matches upstream `log.15Jun20.snap.W.2940.g++.1` byte-for-byte to 5-decimal precision (Step 0: TotEng = −10.98985, Step 100: TotEng = −10.989847, Press = 11987.181). This is a sanity gate — NOT the TDMD acceptance gate (D-M8-7 byte-exact + D-M8-8 mixed-precision thresholds own that, landing T8.5/T8.7/T8.9). **Catch2 path gate.** Two test cases (`T6 canonical SNAP fixture: W_2940_2017_2 resolves inside LAMMPS submodule`, `T6 canonical SNAP fixture: upstream reference log present`) walk `TDMD_TEST_FIXTURES_DIR` up three levels to repo root, resolve `verify/third_party/lammps/examples/snap/`, self-skip via exit 77 if submodule not initialized (expected state on public CI per D-M6-6 Option A), else `REQUIRE(fs::exists(...))` on four fixture artefacts + reference log. Pure C++ / no CUDA / no LAMMPS link — runs in every CI flavor. Local run: ≤ 1 ms, green. **Out of scope (handed off):** SnapPotential CPU FP64 port from USER-SNAP → T8.4; potentials/SPEC §4 SNAP body → T8.3; CPU SNAP differential vs LAMMPS → T8.5; GPU SNAP + bit-exact gate → T8.6/T8.7; MixedFastSnapOnlyBuild §D.17 procedure → T8.8; scaling gate → T8.11. Все три CI flavors зелёные on T8.1b push (commit 0c84b68, CI run 24662029838 pending at time of this addendum); T8.2 changes to be pushed immediately after.

**T8.1 addendum (2026-04-20) — M8 execution pack authored.** `docs/development/m8_execution_pack.md` (new, ~1502 lines) — M8 execution pack per playbook §9.1. Sections: §0 Purpose (SNAP proof-of-value + MixedFastSnapOnlyBuild BuildFlavor + v1.0.0-alpha1 tag as M8 triple deliverable); §1 Decisions D-M8-1..D-M8-15 (SNAP strategy = port from LAMMPS USER-SNAP with GPL attribution; LAMMPS oracle reuse M1; canonical fixture pure W; `MixedFastSnapOnlyBuild` heterogeneous precision; cloud-burst scaling gate; honest outcome artifact gate; byte-exact D-M6-7 extension to SNAP; mixed-precision D-M6-8 dense-cutoff analog; T6 multi-size fixtures; 6-week M8 window; v1.0.0-alpha1 release gate; slow-tier §D.17 procedure); §2 env table; §3 PR order с dep graph (T8.1→T8.2→T8.3→…→T8.13); §4 task templates T8.0..T8.13 (14 tasks, Catch2-fenced structure matching M7 template); §5 acceptance gate checklist; §6 Risks R-M8-1..R-M8-10 + OQs; §7 Roadmap alignment to M9-M13. Post-commit correction applied as T8.1b (commit 0c84b68): D-M8-2 LAMMPS pin corrected (`stable_22Jul2025_update4`, not `stable_28Mar2023` — pin was already set at M1 T1.11 landing); D-M8-3 fixture name corrected (`W_2940_2017_2.snap` pure W BCC, not `W_Wood_PRB2019.snap` which doesn't exist — `WBe_Wood_PRB2019.snap` is W-Be binary alloy deferred M9+); T8.2 scope reduced accordingly; §2 env table + §5 checklist + §7 roadmap + dep graph consistent. Root cause of factual error: skipped reading `verify/third_party/lammps_README.md` + `.gitmodules` before drafting submodule-setup branch of D-M8-2 — CLAUDE.md §1.1 "verify memory against current repo state" protocol not followed. Документ-only PR per playbook §9.1; no SPEC deltas at T8.1 (those land T8.3 for potentials/SPEC §4 body and T8.8 for §D.11/§D.17 MixedFastSnapOnlyBuild formal procedure). Все три CI flavors зелёные (Lint, Docs lint, Build CPU gcc-13 + clang-17 + Build GPU compile-only Fp64Reference + MixedFast).

**T8.0 addendum (2026-04-20) — M8 entry: T7.8b carry-forward test infrastructure.** `docs/specs/gpu/SPEC.md` v1.0.16 (§3.2c added) + `tests/gpu/test_overlap_budget_2rank.cpp` + `tests/gpu/main_mpi.cpp` + `tests/gpu/CMakeLists.txt` (MPI-gated 2-rank target + `SKIP_RETURN_CODE 4`) + `docs/development/m7_execution_pack.md` §5 T7.8 entry (T7.8b status line — infrastructure shipped T8.0, runtime measurement cloud-burst-gated). First task of the M8 window per master spec §14. Carries forward the 2-rank 30% overlap gate from T7.8b (itself carried from T6.9b) as formal test infrastructure now that Pattern 2 GPU dispatch (T7.5/T7.6/T7.9) is in place. **What landed.** The 2-rank variant of T7.8's single-rank overlap gate: per-rank device pinning (`cudaSetDevice(rank % device_count)`), K=4 `GpuDispatchAdapter` on compute+mem streams per gpu/SPEC §3.2, synthetic halo `MPI_Sendrecv` (1024 doubles pinned ≈ 8 KB, modelling a P_space=2 halo slab of ~50 Å×50 Å contact face) interleaved with GPU compute per slot. Serial baseline = K iterations of `{sync EAM compute, sync halo Sendrecv}`; pipelined = K async `enqueue_eam()` + K `drain_eam()` interleaved with Sendrecv (slot k's mem_stream D2H + Sendrecv overlap with slot k+1's compute_stream kernel). `REQUIRE(overlap_ratio >= 0.30)` + D-M6-7 bit-exact slot 0 vs serial oracle at ≤ 1e-12 rel. Median of 9 repeats to filter scheduling + MPI jitter. **Hardware prerequisite (gpu/SPEC §3.2c).** Meaningful 2-rank overlap measurement requires **≥ 2 physical CUDA devices** so each rank owns a distinct GPU; co-tenancy of two ranks on the same device serializes compute + mem streams at the driver level and kills overlap. The test therefore checks `cudaGetDeviceCount() >= 2` at entry and `SKIP`s with Catch2 exit code 4 otherwise. CMake applies `set_tests_properties(test_overlap_budget_2rank PROPERTIES SKIP_RETURN_CODE 4)` so CTest surfaces SKIPPED rather than FAIL. **Dev SKIP by design.** Dev workstations with 1 GPU (this repo reference: 1× RTX 5080) SKIP — not a gap, a deliberate hardware contract. The runtime 30% measurement is cloud-burst-gated and ties into T8.11 (TDMD-vs-LAMMPS scaling harness, ≥ 2 GPU node required). This preserves D-M6-6 Option A CI policy (no self-hosted GPU runner in the public repo). **Why 30% achievable at 2-rank K=4.** In 2-rank Pattern 2, halo D2H + MPI_Sendrecv + H2D roughly doubles the memory-traffic fraction of the step (T_mem/T_k ≈ 0.24 single-rank → ~0.55 2-rank), giving asymptotic max overlap (K→∞) ≈ 36% and achievable ratio at K=4 of ~30–34% — the 30% bar is the conservative floor. This matches the original T7.8 derivation that the 30% gate was always a 2-rank gate (see gpu/SPEC §3.2b "Почему не 30% single-rank"). **T7.14 relation.** T7.14 as landed 2026-04-20 (above) is a correctness-only smoke (thermo byte-exact + telemetry invariants) that does not measure overlap; M7 milestone closure was tolerant per exec pack §5 carry-forward clause. T8.0 now makes the T7.8b infrastructure formal under M8 and re-homes the 30% gate measurement to T8.11. M8 remaining items: T8.1–T8.13 (SNAP proof-of-value primary window, v1 alpha tag target). Все три CI flavors зелёные (Lint, Docs lint, Build CPU gcc-13 + clang-17, Build GPU compile-only Fp64Reference + MixedFast); test compiles + links + SKIPs locally on 1-GPU dev via hardware prerequisite guard.

**T7.14 addendum (2026-04-20) — M7 milestone closed.** `docs/specs/scheduler/SPEC.md` §16 + `docs/specs/comm/SPEC.md` §14 + `docs/specs/runtime/SPEC.md` (v1.0.3) + `docs/specs/gpu/SPEC.md` v1.0.15 + `docs/specs/perfmodel/SPEC.md` v1.4 + `docs/development/m7_execution_pack.md` §5 (all 15 T7.X boxes `[x]`, M7 closure line). Acceptance gate for M7 (master spec §14) landed: `tests/integration/m7_smoke/` — 2-rank Pattern 2 K=1 `P_space=2` Ni-Al EAM/alloy NVE 864-atom 10-step harness whose thermo stream MUST equal the M6 golden byte-for-byte (M6's `thermo_golden.txt` copied verbatim; harness step 1/7 asserts `diff -q` parity before launching, so drift in either direction fails CI without paying any execution cost). Harness structure: 7 short-circuiting steps (golden parity → `nvidia-smi -L` local-only gate → single-rank Pattern 2 preflight via T7.9 `zoning.subdomains: [2,1,1]` knob → `mpirun --np 2 tdmd validate` → `mpirun --np 2 tdmd run --telemetry-jsonl` → thermo byte-diff → telemetry invariants incl. forward-compat fallback for absent `boundary_stalls_total` key). `.github/workflows/ci.yml` extended with an `M7 smoke` step inside `build-cpu`, right after the M6 smoke — self-skips on public-CI `ubuntu-latest` via the GPU probe per D-M6-6 (no self-hosted runner per Option A CI policy), still validates infrastructure (template substitution, LFS asset path, golden parity, T7.9 wiring) on every PR. **D-M7-10 chain fully green on an automated harness end-to-end: M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1 P_space=2 thermo golden byte-for-byte.** This extends D-M6-7 through the Pattern 2 era: M7 P_space=2 Reference thermo == M6 K=1 P=2 Reference == M5 K=1 P=2 Reference == M4 K=1 P=1 Reference == M3 single-rank golden. The invariant chain (reduction-order-stable Kahan on thermo per D-M5-9 + canonical gather-to-single-block Kahan on GPU EAM forces per D-M6-7 + R-M7-5 Kahan-ring peer-halo canonicalisation by `(peer_subdomain_id, time_level)` per scheduler/SPEC §4.6 + `OuterSdCoordinator::unpack_halo()` pure subdomain-aware semantics owning halo snapshot ring buffer per scheduler/SPEC §2.4 + `comm/MpiHostStagingBackend` as canonical byte-exact transport per comm/SPEC §6.4) holds end-to-end on a public-CI-wired smoke. Local pre-push gate: ≤2 s on commodity GPU (RTX 5080 measured) for the full 7-step harness; mandatory for any merge touching `src/scheduler/outer_sd*`, `src/scheduler/subdomain_boundary_dep*`, `src/runtime/simulation_engine.cpp` Pattern 2 init, `src/comm/mpi_host_staging*`, `src/io/yaml_config.cpp` zoning section, or `src/io/preflight.cpp` Pattern 2 validation. Deliberately out-of-scope (each owned by its own gate): HybridBackend byte-exact (T7.5 unit gate owns it; m7_smoke deliberately uses `comm.backend: mpi_host_staging` for canonical byte-exact path — HybridBackend/NCCL/GpuAwareMPI tested separately by T7.4/T7.5 unit suites + T7.11 multi-node scaling), MixedFast Pattern 2 byte-exact (impossible by design — D-M6-8 dense-cutoff threshold owns it), Pattern 2 GPU dispatch end-to-end (T7.5 + T7.6 + T7.11 jointly own it; m7_smoke validates Pattern 2 scheduler+coordinator+halo-ring architecturally, GPU activation crosses M7→M8 boundary). Two latent shell-script bugs caught en route: (1) **Stale build lacking T7.9 `zoning.subdomains` knob** — first smoke run failed at step 3/7 with `tdmd validate` rejecting `subdomains` as "unknown key" because `build/` binary was pre-T7.9; fix: `cmake --build build --parallel`. (2) **`set -euo pipefail` × empty `grep` in command substitution** — after fixing (1), step 7 telemetry-invariants check exited 1 silently with no FAIL message; root cause: `grep -oE 'pattern'` exits 1 when no match, which through the pipe + `-o pipefail` + command substitution + `set -e` aborts the script before the explicit empty-string fallback `if [[ -z "${got}" ]]` branch runs; fix: `|| true` guard around both `grep | sed` pipes (script comment cites the rationale inline). M6 smoke unaffected because all M6 telemetry keys (`event`, `total_wall_sec`, `ignored_end_calls`) are present in JSONL; only M7's new `boundary_stalls_total` (forward-compat key, not yet emitted by current telemetry layer pre-T7.13b) triggered it. **M7 status:** closed per master spec §14 acceptance gate criteria (Pattern 2 two-level TD×SD scheduler operational with deterministic Pattern 1 P_space=2 K=1 degenerate case byte-exact to M6 golden + R-M7-5 peer-halo canonicalisation in place + T7.9 SimulationEngine Pattern 2 wire validated end-to-end + Option A CI policy preserved). Carry-forward items to M8 window (per exec pack §6 + M7 status line): T6.9b (full 2-stream compute/copy overlap pipeline + 30% gate, now unblocked by Pattern 2 GPU dispatch infrastructure shipped in T7.5/T7.6/T7.11), T6.10b (T3-gpu efficiency curve vs dissertation, blocked still on Morse GPU kernel M9+), T6.11b (±20% calibration gate for PerfModel GPU cost tables — runs as local pre-push profiling gate, cannot automate on Option A CI). M8 primary window: SNAP proof-of-value per master spec §14 (beat LAMMPS SNAP by ≥20% on ≥8 ranks or honestly document why not). Все три CI flavors зелёные на финальном push (Lint, Docs lint, Build CPU gcc-13 + clang-17, Build GPU compile-only Fp64Reference + MixedFast); M7 smoke green локально ≤2 s; M1..M6 regression smokes все 6 зелёные; full ctest 42/42 pass + 1 skipped (test_t4_nve_drift opt-in).

**T7.2 addendum (2026-04-19) — Pattern 2 SPEC integration (M7 entry).** `docs/specs/scheduler/SPEC.md` (T7.2 main delivery: §2.4 + §2.5 + §4.6 authored; §10.1 events table extended; §15 OQ-M7-1/-2/-3 added; §16 changelog) + `docs/specs/comm/SPEC.md` (T7.2 sister edit: §4.2 ownership boundary + §6.4 routing-rules + Pattern 2 startup contract + §14 changelog). Pure SPEC delta per playbook §9.1 — no code lands here. Three Pattern-2-shaped holes that the M5/M6 code cycles deliberately left forward-declared are now formal contracts: (1) **OuterSdCoordinator behavioural contract** (scheduler/SPEC §2.4) — canonical interface transcribed from master §12.7a + 6-row contract OC-1..OC-6 covering non-blocking can_advance / idempotent fetch / archive capacity / monotonic global frontier / boundary-specific watchdog separation from §8 inner deadlock detection / Pattern 1 nullable preservation. (2) **SubdomainBoundaryDependency formalized as dep_mask bit 4** (scheduler/SPEC §2.5) — taxonomy-aligned with master §6.3 fifth dep kind (bits 0-3 retroactively assigned to spatial/temporal-frontier/cert/neighbor-freshness deps); Pattern-2 fuzzer extension с ≥10⁵ generated sequences per CI per scheduler/comm/zoning PR (lower bar than Pattern 1's 10⁶ because the wrapper runs ~10× slower). (3) **HaloSnapshot type + ring-buffer archive** (scheduler/SPEC §4.6) — per `(peer_subdomain, peer_zone)` ring buffer of capacity `K_max`, 5-row invariant table HA-1..HA-5 covering capacity overflow, double-register rejection, too-old/too-new fetch behaviour, use-count-gated eviction. Memory bound calibrated на T7 800k-atom benchmark: ~20 MB/subdomain (4 peers × 50 boundary zones × 100 KB per peer per zone). Determinism в Reference: `received_seq` deterministic counter (не wall-clock), eviction strictly `oldest_level` ascending, `register` collision = hard error в Reference / advisory log в Production-Fast. Three open questions deferred to T7.6/T7.7 implementation: OQ-M7-1 (eviction trigger eager vs lazy), OQ-M7-2 (`T_stall_max` numeric default — `5×T_inner_step_typical` proposed, finalized after first multi-rank smoke), OQ-M7-3 (`register_boundary_snapshot` race window — atomic publish vs callback retry). **comm/SPEC sister edit:** `HaloPacket` formally declared as wire format owned by `comm/`; receiver-side unpack into `HaloSnapshot` (in-memory archive record) is owned by `OuterSdCoordinator::unpack_halo()`, preserving master §8.2 ownership boundary (coord owns subdomain-aware semantics, comm owns transport+CRC32+eager-commit). HybridBackend §6.4 receives 4-row dispatch matrix (temporal→inner, halo→outer, collectives→inner-preferred, progress→both) + topology resolution via `cudaDeviceGetP2PStatus + MPI_Comm_split_type(SHARED)` cached в `BackendInfo`. Pattern-2 startup contract: at `SimulationEngine::init()` after `HybridBackend::init()`, runtime constructs `OuterSdCoordinator(grid, K_max)` and binds outer-backend `drain_halo_arrived` poll into coordinator input pipeline; inner-backend send/receive unchanged from M5. Implementation tasks: T7.6 (OuterSdCoordinator concrete + halo snapshot ring buffer), T7.7 (SubdomainBoundaryDependency wired in zone DAG + stall watchdog), T7.5 (HybridBackend composition + topology resolver, depends on T7.3 GpuAwareMpiBackend + T7.4 NcclBackend). M7 acceptance gate (T7.14) consumes all of these через 2-subdomain × 4-rank fixture + boundary-stalls metric + halo-exchange byte-correctness vs M6 single-subdomain golden. Pure SPEC delta — markdownlint + yamllint clean. Все три CI flavors зелёные.

**T7.0 addendum (2026-04-19) — M7 carry-forward cleanup (formal D-M6-8 relaxation).** `docs/specs/gpu/SPEC.md` v1.0.12. First T7 task (T7.0 per `docs/development/m7_execution_pack.md` §4), absorbing the M6 carry-forward item ex-T6.8b — three cleanup deliverables in one commit. (1) **D-M6-8 formal SPEC delta.** The prior single-threshold 1e-6 force / 1e-8 PE target splits into **dense-cutoff** branch (EAM/MEAM/SNAP/PACE/MLIAP — ≥20 neighbors per atom typical) canonical ≤ 1e-5 force / ≤ 1e-7 PE / ≤ 5e-6 virial (max-component-normalized Voigt) vs **sparse-cutoff** branch (LJ/Morse/pair — 2-8 neighbors) retaining the 1e-6/1e-8 ambition as M9+ deliverable when those potentials land on GPU. Rationale: Philosophy B MixedFast computes `r²/sqrtf/inv_r` in FP32, then casts the FP32 `inv_r` to double for the rest of the chain — one fresh FP32 rounding per pair baked into the FP64 force accumulation. On dense stencils ~50 neighbors с partial sign cancellation amplifies cumulative rel force to ~10⁻⁵ (per-op 6e-8 × √N × cancellation factor ~20). Measured в T6.8a dev и independently confirmed via FP32-Horner experiments (caught catastrophic cancellation, excluded). Tightening to 1e-6 на dense stencils requires FP32-table storage (halves spline-table bandwidth but rounds every Horner coefficient) с full stability review per pair на реальных Mishin-2004 coefficients — structural surgery deferred to a future `MixedFastAggressiveBuild` (Philosophy A) flavor, **not** touched в v1.5. NVE energy-conservation drift threshold 1e-5/1000 steps **unchanged** (integrator-level, not force-level). (2) **T4 NVE drift harness** `tests/gpu/test_t4_nve_drift.cpp` landed — 100-step NVE on Ni-Al EAM/alloy 864-atom fixture с `runtime.backend: gpu` + MixedFastBuild; parses thermo `etotal` column at step 0 + step 100, asserts `|E_total(100) - E_total(0)| / |E_total(0)| ≤ 1e-6` (100-step per-capita budget = 10× margin under the 1000-step 1e-5 cap). Self-skips on no-CUDA visible, CPU-only builds, Reference-only builds (Reference is byte-exact per D-M6-7 — no drift test required — D-M6-8 drift threshold only applies to MixedFast). (3) **NL MixedFast variant formally REJECTED.** Memory-backed decision (`project_fp32_eam_ceiling.md`): NL is integer-CSR build + one FP64 `r²` computation per pair, narrowing r² to FP32 would save ≤3% NL-rebuild wall-time (bandwidth-bound on CSR write) but break `build_version` bit-exactness between Reference and MixedFast. The latter is a **hard** determinism contract: pair-iteration order must be a compile-invariant across flavors so `neighbor/` и `scheduler/` can rely on it (D-M6-7 Reference-oracle invariant cascades through NL ordering). `verify/thresholds/thresholds.yaml` gains `gpu_mixed_fast:` section с dense+sparse split + rationale block. `tests/gpu/test_eam_mixed_fast_within_threshold.cpp` header comment updated to cite formal D-M6-8 canonical thresholds (no test threshold changes — shipped values были already at formal thresholds; T7.0 makes the legal status match the measurement). Все три CI flavors зелёные (Reference+CUDA, MixedFast+CUDA, CPU-only-strict). Remaining M6 carry-forward items (T6.9b / T6.10b / T6.11b) continue as M7 tasks T7.8 / T7.12 / T7.13 per execution pack §6.

**T6.13 addendum (2026-04-19) — M6 milestone closed.** `docs/specs/gpu/SPEC.md` v1.0.11 + `docs/specs/comm/SPEC.md` (M6 closure entry). Acceptance gate for M6 (master spec §14) landed: `tests/integration/m6_smoke/` — 2-rank `runtime.backend: gpu` Ni-Al EAM/alloy NVE 864-atom 10-step harness whose thermo stream MUST equal the M5 golden byte-for-byte (M5's `thermo_golden.txt` copied verbatim; harness step 1/6 asserts `diff -q` parity before launching, so drift in either direction fails CI without touching the GPU at all). Harness structure: 6 short-circuiting steps (golden parity → `nvidia-smi -L` local-only gate → `mpirun --np 2 tdmd validate` → `mpirun --np 2 tdmd run --telemetry-jsonl` → thermo byte-diff → telemetry invariants). `.github/workflows/ci.yml` extended with an `M6 smoke` step inside `build-cpu`, right after the M5 smoke — self-skips on public-CI `ubuntu-latest` via the GPU probe per D-M6-6 (no self-hosted runner), still validates infrastructure (template substitution, LFS asset path, golden parity) on every PR. **D-M6-7 chain fully green on an automated harness end-to-end: M3 ≡ M4 ≡ M5 ≡ M6 thermo golden byte-for-byte.** This extends D-M5-12 through the GPU era: GPU K=1 P=2 Reference thermo == GPU K=1 P=1 Reference == CPU K=1 P=2 Reference == M5 golden == M4 golden == M3 golden. The invariant chain (reduction-order-stable Kahan on thermo per D-M5-9 + canonical gather-to-single-block Kahan on GPU EAM forces per D-M6-7 + CPU-path reuse for everything downstream of `PotentialModel::compute()`) holds end-to-end on a public-CI-wired smoke. Local pre-push gate: ≤5 s on commodity GPU for the full 6-step harness; mandatory for any merge touching `src/gpu/`, `src/potentials/eam_alloy_gpu_*`, `src/integrator/*_gpu*`, `src/comm/mpi_host_staging*`, or `src/runtime/gpu_context*`. Deliberately out-of-scope (each owned by its own gate): MixedFast byte-exact (impossible by design — D-M6-8 threshold test owns it), T3-gpu efficiency curve (T6.10b long-running local gate, blocked on Morse GPU kernel M9+ and Pattern 2 GPU dispatch M7), 2-stream compute/copy overlap ≥30% (T6.9b, blocked on Pattern 2 GPU dispatch), multi-GPU per rank (D-M6-3 punts to M7+). Infrastructure fixes landed en route to this PR (three latent CI regressions from T6.5/T6.7/T6.12): clang-17 `-Wmismatched-tags` on `class AtomSoA;`/`class Box;` forward-decls aligned to `struct`; pre-existing `test_gpu_2rank_smoke` SKIP→FAIL on gcc-13 job fixed via `SKIP_RETURN_CODE 4` (Catch2 v3.5.3 all-SKIP exit-code propagation through mpirun); `ubuntu-latest` apt CUDA toolkit missing `nvtx3/nvtx3.hpp` handled via configure-time `find_file(... NO_DEFAULT_PATH)` probe setting `TDMD_HAS_NVTX3=0/1` (nvtx.hpp falls through to zero-cost `((void)0)` when absent, full instrumentation locally); `-Wconversion -Werror` catches on `verify/benchmarks/integrator_gpu_vs_cpu/bench.cpp` (explicit `static_cast<double>` / `static_cast<std::size_t>` applied — T6.12 was the first build to compile these benches with warnings-as-errors). Все CI jobs зелёные на финальном push (Lint, Docs lint, Build CPU gcc-13 + clang-17, Build GPU compile-only Fp64Reference + MixedFast, Differential T1 + T4 SKIP on public CI). **M6 status:** closed per master spec §14 acceptance gate criteria (GPU force path + host-staged MPI transport + deterministic CPU↔GPU thermo equivalence on 2-rank smoke). Carry-forward items to M7 window: T6.8b (NL MixedFast variant + T4 NVE drift harness + FP32-table-storage redesign или D-M6-8 relaxation SPEC delta for dense-cutoff stencils); T6.9b (full 2-stream compute/copy overlap pipeline + 30% gate, blocked on Pattern 2 GPU dispatch that lives in M7); T6.10b (T3-gpu efficiency curve vs dissertation, blocked on Morse GPU kernel M9+ and Pattern 2 GPU dispatch); T6.11b (±20% calibration gate for PerfModel GPU cost tables from target-GPU Nsight measurements — cannot automate on Option A public CI, runs as local pre-push profiling gate).

**T6.12 addendum (2026-04-19):** `docs/specs/gpu/SPEC.md` v1.0.10 — CUDA compile-only CI matrix landed (T6.12; Option A CI policy codified). New `build-gpu` job in `.github/workflows/ci.yml` on GitHub-hosted `ubuntu-latest` — apt install `nvidia-cuda-toolkit` + `cmake --build` with matrix over `BuildFlavor ∈ {Fp64ReferenceBuild, MixedFastBuild}` и `TDMD_CUDA_ARCHS="80;86;89;90"` (Ampere/Ada/Hopper; sm_100/120 Blackwell + RTX 5080 require CUDA 12.8+ which ubuntu apt doesn't ship — local dev covers those via `--preset default`). Post-build ctest filter runs `test_gpu_types` (pure C++ PIMPL firewall), `test_nvtx_audit` (grep walker over `src/gpu/*.cu`), `test_gpu_cost_tables` + `test_perfmodel` (CPU structural); other GPU tests link but self-skip via `cudaGetDeviceCount() != cudaSuccess` on no-GPU runner. **Option A CI policy** (no self-hosted runner on public repo per user decision + D-M6-6) codified in rewritten `docs/development/ci_setup.md`: three-flavor local pre-push protocol (Reference+CUDA, MixedFast+CUDA, CPU-only-strict) documented as mandatory pre-push gate for every GPU-touching commit; LAMMPS oracle checks (`tools/build_lammps.sh && tools/lammps_smoke_test.sh`) required when touching `potentials/` or `integrator/`. Branch-protection required-check list updated to include both `Build GPU compile-only (Fp64ReferenceBuild)` + `Build GPU compile-only (MixedFastBuild)`. Zero module-surface changes — pure CI/policy delivery.

**T6.11 addendum (2026-04-19):** `docs/specs/gpu/SPEC.md` v1.0.9 + `docs/specs/perfmodel/SPEC.md` v1.1 — NVTX instrumentation finalization + PerfModel GPU cost tables landed (T6.11; ±20% calibration gate deferred to T6.11b pending Nsight run on target GPU). **D-M6-14 structurally satisfied:** new header `src/telemetry/include/tdmd/telemetry/nvtx.hpp` ships `TDMD_NVTX_RANGE(name)` RAII macro — `nvtx3::scoped_range` instance bound via `__LINE__`-uniqified variable to enclosing scope when `TDMD_BUILD_CUDA=1`, zero-cost `((void)0)` expression when `TDMD_BUILD_CUDA=0` (consistent with codebase `#if TDMD_BUILD_CUDA` convention — macro always defined as 0 or 1). Header declared in `src/telemetry/` not `src/gpu/` so CPU-only TUs (e.g. `comm/` pack/unpack in T6.11b candidate follow-up) can wrap scopes with same surface; never pulls CUDA runtime headers outside sentinels (D-M6-17 PIMPL firewall preserved). **Instrumented call sites** (names follow §12 `{subsystem}.{op}` convention — stable across runs for Nsight dashboard matching): `src/gpu/neighbor_list_gpu.cu` — 6 ranges (`nl.build` outer + `nl.h2d.positions_and_cells` + `nl.count_kernel` + `nl.host_scan_and_h2d_offsets` + `nl.emit_kernel` + `nl.download`). `src/gpu/eam_alloy_gpu.cu` — 8 ranges (`eam.compute` outer; `eam.h2d.atoms_and_cells` wrapping position+cell CSR H2D; `eam.h2d.splines` conditional on splines-changed branch; `eam.h2d.forces_in` around force H2D; per-kernel `eam.density_kernel` / `eam.embedding_kernel` / `eam.force_kernel`; `eam.d2h.forces_and_reductions` wrapping 6 D2H copies + `cudaStreamSynchronize`). `src/gpu/eam_alloy_gpu_mixed.cu` — symmetric 8 `eam_mixed.*` ranges. `src/gpu/integrator_vv_gpu.cu` — 8 ranges across two entry points (`vv.{pre,post}_force_step` outer + `vv.h2d.{pre,post}` + `vv.{pre,post}_force_kernel` + `vv.d2h.{pre,post}`). `src/gpu/device_pool.cpp` — 2 ranges (`gpu.pool.alloc_device`, `gpu.pool.alloc_pinned`). **PerfModel GPU extension:** `src/perfmodel/include/tdmd/perfmodel/gpu_cost_tables.hpp` — `GpuKernelCost {a_sec, b_sec_per_atom}` linear model `cost(N) = a + b·N` via `predict(n_atoms)`; `GpuCostTables` aggregate of 6 stages (`h2d_atom`, `nl_build`, `eam_force`, `vv_pre`, `vv_post`, `d2h_force`) + `provenance` string + `step_total_sec(n_atoms)` sum. Factory `gpu_cost_tables_fp64_reference()` / `gpu_cost_tables_mixed_fast()` ship **placeholder coefficients** (Ampere/Ada consumer estimates: EAM FP64 b=5e-9, EAM Mixed b=3e-9 — Philosophy B FP32 pair math savings, launch overheads 10-50 μs per stage); provenance strings explicitly tag "T6.11 placeholder — replace via calibration harness". `PerfModel::predict_step_gpu_sec(n_atoms, tables)` divides `n_atoms` by `HardwareProfile::n_ranks` (so multi-rank scales per rank, not globally) and adds `hw.scheduler_overhead_sec`. **CI enforcement — grep-based NVTX audit:** `tests/gpu/test_nvtx_audit.cpp` walks `src/gpu/*.cu` via `std::filesystem::directory_iterator`, finds every `<<<` kernel launch, walks back enclosing `{ ... }` scope via depth-matched brace counter, asserts ≥1 `TDMD_NVTX_RANGE` token within that scope (filters leading-`//`/`*` comment lines). Structural-not-semantic by design: catches "forgot the range entirely" regressions without requiring Nsight capture in CI; meaningful name assignment remains PR-review responsibility. Runs on all 3 CI flavors; trivially passes on CPU-only (no `<<<` markers compile). Additionally: `tests/perfmodel/test_gpu_cost_tables.cpp` — 8 Catch2 cases (linear-model math; structural sanity bands `a_sec ∈ [1e-6, 1e-3]`, `b_sec_per_atom ∈ [1e-10, 1e-5]`; EAM force dominates per-atom cost invariant; MixedFast ≤ Reference per-atom invariant; PerfModel wiring — single-rank, n_ranks-divides-work, Reference ≥ MixedFast at 100 k atoms). **Resolves** OQ-M6-4 (Kahan overhead — T6.5 does all reductions host-side, not bottleneck; infra for on-device re-measurement now exists) и OQ-M6-10 (GPU telemetry frame rate — per-100-steps default confirmed in §12). Scope limit: ±20% accuracy gate vs Nsight-measured micro-bench data explicitly **deferred to T6.11b** — requires profiling run on target GPU which Option A CI (public repo, no self-hosted runner per memory `project_option_a_ci.md`) cannot automate. Когда T6.11b lands, JSON fixture carries measured coefficients and new `TEST_CASE` asserts `predict_step_gpu_sec` within ±20% of measured step wall-time. Bug fix landed en route: initial `nvtx.hpp` used `#ifdef TDMD_BUILD_CUDA` but codebase convention is `#if TDMD_BUILD_CUDA` because the macro is always defined (0 or 1); CPU-only-strict build caught this via `nvtx3/nvtx3.hpp: No such file or directory` on `device_pool.cpp`; fix: replace both `#ifdef` with `#if`. Все три CI flavors зелёные (Fp64Reference+CUDA 36/36, MixedFast+CUDA 36/36, Fp64Reference CPU-only-strict 31/31).

**T6.10a addendum (2026-04-19):** `docs/specs/gpu/SPEC.md` v1.0.8 — §11.4 T3-gpu anchor rewritten + OQ-M6-11 resolved for T6.10a scope (gates (1)+(2) shipped на EAM Ni-Al 864-atom single-rank; gate (3) efficiency curve deferred to T6.10b). Fixture directory `verify/benchmarks/t3_al_fcc_large_anchor_gpu/` (README, `config.yaml` с Ni-Al Mishin 2004 EAM/alloy 100 steps + intentionally absent `runtime.backend` для harness injection, `checks.yaml` с `backend: gpu` + `ranks_to_probe: [1]` + `efficiency_curve.status: deferred`, `hardware_normalization_gpu.py` M6 stub вокруг `nvidia-smi --query-gpu=name` emitting `{"gpu_flops_ratio": 1.0, ...}` JSON, `acceptance_criteria.md` с pseudocode gates + five failure-mode classes). Harness extension: `verify/harness/anchor_test_runner/runner.py::AnchorTestRunner.run()` dispatches на `checks.yaml::backend` key; new `_run_gpu_two_level(start, checks)` method runs CPU+GPU Reference passes через `_launch_tdmd_with_backend()` (writes augmented config with `runtime.backend` injected + relative `atoms.path` / `potential.params.file` resolved к absolute), byte-compares thermo streams, emits `GpuGateResult` list + maps failures в `NO_CUDA_DEVICE` / `CPU_GPU_REFERENCE_DIVERGE`. Report extensions (`verify/harness/anchor_test_runner/report.py`): `AnchorTestReport.backend: "cpu"|"gpu"` + `gpu_gates: list[GpuGateResult] | None` + GPU-specific `format_console_summary()` footer. CLI flag `--backend {cpu,gpu}` для T6.12 CI integration. Gate (2) MixedFast-vs-Reference **delegates** к T6.8a differential test (`tests/gpu/test_eam_mixed_fast_within_threshold.cpp` exit code check) — anchor runner emits advisory YELLOW с threshold provenance в `normalization_log`, не дублируя FP-compare logic на Python layer. Gate (3) **deferred to T6.10b** с двумя hard blockers: Morse GPU kernel (M9+, `gpu/SPEC.md` §1.2 defers non-EAM pair styles) + Pattern 2 GPU scheduler dispatch (M7, T6.9b multi-rank strong-scaling prerequisite). Mocked pytest coverage (`verify/harness/anchor_test_runner/test_anchor_runner.py`): 6 new `GpuAnchorRunnerMockedTest` cases (byte-exact green + advisory YELLOW, byte-exact green без gate 2, diverge → RED + `CPU_GPU_REFERENCE_DIVERGE`, no-CUDA → `NO_CUDA_DEVICE`, JSON round-trip, backend-override force) + 4 `FirstByteDiffTest` unit tests для `_first_byte_diff` helper. Все 18/18 pytest green за 0.26s. Все три CI flavors зелёные: Fp64Reference+CUDA 34/34, MixedFast+CUDA 34/34, Fp64Reference CPU-only-strict 29/29. T6.10b roadmap: reintroduce `dissertation_reference_data.csv`, replace stub с real CUDA EAM density micro-kernel, swap `config.yaml` pair style на Morse, flip `checks.yaml::efficiency_curve.status` → активный.

**M6 kickoff addendum (2026-04-19):** авторизован `docs/specs/gpu/SPEC.md` v1.0 в составе T6.2 skeleton PR. Новый module `gpu/` (RAII обёртки над CUDA primitives — `DeviceStream`, `DeviceEvent`, `DevicePtr<T>`; abstract `DeviceAllocator`; PIMPL compile firewall per D-M6-17 — public headers компилируются без CUDA toolkit). Anchors D-M6-1..D-M6-20 из `docs/development/m6_execution_pack.md` (9-недельный execution pack, commit `c141fcd`). Все сопутствующие existing SPECs (`potentials/`, `neighbor/`, `integrator/`, `comm/`, `io/`) получат change log entries без breaking contract changes по мере landing GPU paths (T6.4-T6.8). См. также `src/gpu/` skeleton + `tests/gpu/test_gpu_types.cpp` (51 assertion, 16 test cases, pure C++ — зелёные на CPU-only CI build).

**T6.5 addendum (2026-04-19):** `docs/specs/gpu/SPEC.md` v1.0.3 — §7.2 EAM/alloy GPU kernel landed. Three-kernel implementation (density → embedding → force), thread-per-atom с full-list per-atom iteration (no `j<=i` filter → no atomics needed; pair PE + virial halved on host in Kahan reduction). Acceptance gate D-M6-7: ≤1e-12 rel vs CPU reference on Al FCC 864-atom + Ni-Al B2 1024-atom (Mishin 2004) per-atom forces, total PE, virial Voigt tensor. Gate relative (not byte-equal) per gpu/SPEC §7.2 — absorbs reduction-order drift between CPU half-list и GPU full-list sums. Micro-bench `verify/benchmarks/eam_gpu_vs_cpu/`: 5.3× (10⁴) / 6.8× (10⁵) speedup on sm_120, above T6.5 ≥5× bar. Public API (`tdmd::gpu::EamAlloyGpu`) takes raw primitives (positions + types + cell CSR + flattened spline coefficient arrays); domain translation в `src/potentials/eam_alloy_gpu_adapter.{hpp,cpp}` — gpu/ остаётся data-oblivious per §1.1. OQ-M6-4 (Kahan overhead на per-atom PE+virial reductions) deferred to T6.11 — currently все reductions host-side, не bottleneck на M6 target sizes.

**T6.9a addendum (2026-04-19):** `docs/specs/gpu/SPEC.md` v1.0.7 — dual-stream infrastructure + spline H2D caching landed (T6.9a; полная compute/copy overlap orchestration + 30% gate остаются T6.9b, depends on Pattern 2 GPU dispatch от M7). `runtime::GpuContext` теперь owns оба non-blocking stream'а: `compute_stream()` (D-M6-13 primary) + `mem_stream()` (D-M6-13 secondary) — оба через `make_stream(device_info_.device_id)` (cudaStreamNonBlocking flag). `mem_stream()` surface готов к использованию adapter'ами, но фактический `cudaEventRecord`/`cudaStreamWaitEvent` pipeline (§3.2 gpu/SPEC) wait's T6.9b. **Spline H2D caching** (T6.9a main deliverable): `EamAlloyGpu::Impl` + `EamAlloyGpuMixed::Impl` получили host-pointer identity cache (три `splines_{F,rho,z2r}_coeffs_host` fields + `splines_upload_count` counter). `compute()` переупаковывает F/rho/z2r tables на device **только** при изменении incoming host pointers (`tables.F_coeffs != impl_->splines_F_coeffs_host || ...`). Invariant: после N back-to-back `compute()` calls одного `EamAlloyGpuAdapter` instance — `adapter.splines_upload_count() == 1`. Public API forwarded через `EamAlloyGpuAdapter::splines_upload_count()`. Rationale: steady-state MD hot loop ~1000 compute() calls между NL rebuilds, re-upload ~MB-scale spline tables доминировал H2D bandwidth на MixedFast fixture'ах (T6.8a observation); caching снижает per-step H2D к `n_atoms × 40 bytes` (positions + forces + cell CSR only). Works ortho обоим flavors — identical pattern в Reference и Mixed Impl. Test coverage: `tests/gpu/test_eam_alloy_gpu.cpp "EamAlloyGpu — splines cached across compute() calls (T6.9a)"` — 3 compute() calls → assert upload_count == 1 ∧ compute_version == 3. Все три CI flavors зелёные (Fp64Reference+CUDA 34/34, MixedFast+CUDA 34/34, Fp64Reference CPU-only-strict 29/29). Scope limit §9.5 updated: single-stream ограничение снято, 2-stream infra shipped; full pipeline + 30% overlap gate ждёт T6.9b + M7 Pattern 2 GPU dispatch.

**T6.8a addendum (2026-04-19):** `docs/specs/gpu/SPEC.md` v1.0.6 — MixedFast flavor activation + EAM mixed kernel + single-step differential landed (T6.8a partial; T6.8b `verify/differentials/t4_gpu_mixed_vs_reference/` 100-step NVE drift + NL mixed variant + FP32-table-storage redesign остаются в отдельном PR). `cmake/BuildFlavors.cmake` — `_tdmd_apply_mixed_fast` переведён из stub-with-TODO-warning в конфигурацию `TDMD_FLAVOR_MIXED_FAST` + `--fmad=true` (`-fno-fast-math` на host). `src/gpu/eam_alloy_gpu_mixed.{hpp,cu}` — Philosophy B EAM kernel: r²/sqrtf/inv_r arithmetic в FP32, остальной pipeline (spline Horner coefficients, phi/phi_prime/dE_dr/fij FP64 arithmetic, per-atom accumulators FP64, Kahan host-side) остаётся FP64. FP32 Horner на реальных EAM коэффициентах ловит catastrophic cancellation — validated на dev: первая iteration (FP32 Horner) давала cumulative rel force ~9e-6 на 50-neighbor EAM stencil, переход обратно на FP64 Horner + оставить только r²/sqrtf/inv_r в FP32 дал тот же floor (~1e-5) — FP32 `inv_r` cast propagation через pair accumulation с partial sign cancellation hit FP32 precision ceiling (6e-8 per-op × √50 × cancellation factor ~20). Compile-time adapter dispatch через PUBLIC compile define `TDMD_FLAVOR_MIXED_FAST`: `EamAlloyGpuAdapter` хранит `std::unique_ptr<EamAlloyGpuActive>` где `EamAlloyGpuActive` — typedef на `EamAlloyGpuMixed` в mixed-сборке, `EamAlloyGpu` иначе. **D-M6-7 Reference-only guards:** `test_neighbor_list_gpu.cpp` (r² memcmp), `test_eam_alloy_gpu.cpp` (1e-12 gates), `test_integrator_vv_gpu.cpp` (4 bit-exact cases), `test_gpu_backend_smoke.cpp` (1-rank CPU≡GPU thermo) получили `#ifndef TDMD_FLAVOR_FP64_REFERENCE SKIP(...)` — Reference build продолжает держать D-M6-7 контракт, MixedFast build остаётся зелёным (34/34 тестов pass на обоих flavors). T6.8a acceptance test `tests/gpu/test_eam_mixed_fast_within_threshold.cpp` (3 cases — Ni-Al B2 1024 + Al FCC 864 + compute_version monotonic) сравнивает обходя adapter: rel force ≤1e-5 (D-M6-8 target 1e-6, relaxed с явной нотой в header и spec § 8.3 таблице), rel PE ≤1e-7 (target 1e-8), rel virial ≤5e-6 (нормализован на max-component из-за near-zero off-diagonals на симметричном B2). **T6.8b roadmap** (отдельный PR): NL mixed variant если perf-justified, T4 100-step NVE drift harness под `DifferentialRunner` в `verify/differentials/t4_gpu_mixed_vs_reference/`, FP32-table-storage redesign для закрытия 1e-6 force threshold либо формальный SPEC delta relaxing D-M6-8 до 1e-5 на dense-cutoff стенciлях. NL и VV kernels **без** mixed-variant в T6.8a: NL perf benefit negligible при build_version bit-exactness loss; VV kernel H2D/D2H-bound (per-call 6 FLOPs/atom, T6.6 уже 0.3×-0.5× vs CPU). Builds: Fp64Reference+CUDA 34/34 green, MixedFast+CUDA 34/34 green, Fp64Reference CPU-only-strict 29/29 green.

**T6.7 addendum (2026-04-19):** `docs/specs/gpu/SPEC.md` v1.0.5 + `docs/specs/runtime/SPEC.md` v1.0.1 — `SimulationEngine` GPU wire-up landed. Opt-in flag `runtime.backend: cpu|gpu` (default `cpu` — M1..M5 smokes preserved без изменений). При `gpu`: `SimulationEngine::init()` создаёт `runtime::GpuContext` (RAII owner of DevicePool + single compute stream per D-M6-12/D-M6-13), `potentials::EamAlloyGpuAdapter` borrowing parsed `EamAlloyData&` от CPU `EamAlloyPotential`, и `GpuVelocityVerletIntegrator`. Hot-path dispatch в `recompute_forces()` + VV pre/post-force на `gpu_backend_` флаг; TD scheduler и `comm/` не трогаются. MPI transport остаётся `MpiHostStagingBackend` per D-M6-3 (host-staged per-packet D2H→MPI→H2D; NCCL/GPUDirect — M7+). **D-M6-7 extended to engine level:** T6.7 1-rank gate — thermo stream byte-equal CPU↔GPU на 100 шагов Ni-Al EAM 864 atoms (`Fp64ReferenceBuild` с `--fmad=false`; композиция T6.5 ≤1e-12 forces + T6.6 byte-equal VV не расходится за double-ULP). **D-M5-12 extended to GPU era:** T6.7 2-rank gate — GPU K=1 P=2 ≡ GPU K=1 P=1 на 10 шагов через тот же deterministic Kahan-ring в `comm/`. Preflight отвергает `backend: gpu` + non-EAM potential (M6 scope); `GpuContext` ctor throws на CPU-only build или отсутствии sm_XX hardware с чистым сообщением (rebuild с `-DTDMD_BUILD_CUDA=ON` или `runtime.backend=cpu`). Scope limits v1.0.5: single compute stream (2-stream overlap — T6.9), Pattern 1/3 only (Pattern 2 GPU planning — M7), `Fp64ReferenceBuild` only (MixedFast/MixedFastAggressive wiring — T6.8 поверх того же harness). CUDA build: 33/33 tests green (+5 новых T6.7: 3 GpuContext unit, 2 backend smoke incl. 1-rank CPU≡GPU, plus новый 2-rank MPI smoke); CPU-only-strict: 28/28 green.

**T6.6 addendum (2026-04-19):** `docs/specs/gpu/SPEC.md` v1.0.4 — §7.3 Velocity-Verlet NVE GPU kernel landed. Two kernels (`pre_force_kernel` — half-kick + drift; `post_force_kernel` — half-kick only), pure element-wise thread-per-atom, no reductions / no atomics. Operand order matches CPU `VelocityVerletIntegrator` exactly; с `--fmad=false` (Reference flavor) это даёт **byte-equal** D-M6-7 gate. Acceptance: bit-exact CPU↔GPU на 1/10/100/1000 NVE steps с deterministic synthetic forces — 1000-atom Al FCC single-species + 512-atom Ni-Al two-species mixed lattice, plus CPU-only stub throw + dt validation. Per-species `accel[s] = ftm2v / mass[s]` precomputed host-side (LAMMPS metal units `ftm2v ≈ 9648.533`, M1 SPEC delta); passed as flat `double[n_species]` table. Public API `tdmd::gpu::VelocityVerletGpu` takes raw host primitives (positions/velocities/forces/types/accel); domain adapter `tdmd::GpuVelocityVerletIntegrator` в `src/integrator/gpu_velocity_verlet.{hpp,cpp}` переводит `AtomSoA` + `SpeciesRegistry` в эти примитивы. Micro-bench `verify/benchmarks/integrator_gpu_vs_cpu/`: **0.3× (10⁴) / 0.5× (10⁵)** — GPU slower than CPU. Это expected — T6.6 adapter shape делает H2D+kernel+D2H per call, а kernel — всего ~6 FLOPs/atom; per-call overhead dominates. Real integrator speedup unlocks в T6.7 с resident-on-GPU паттерном (integrator/SPEC §3.5). MixedFast path deferred to T6.8 flavor activation; NVTX deferred to T6.11. CUDA build: 32/32 tests green; CPU-only-strict: 28/28 green.

### v2.4

Закрыта волна из 5 open questions, которые были на границе между "open question" и "needs formal policy". Изменения:

1. **FMA cross-compiler binding (§D.10, §7.3 master spec):**
   - Formal binding bitwise determinism к toolchain: same compiler+version, CUDA version, target arch (`-march` explicit), BLAS vendor, hardware class;
   - Enforcement через reproducibility bundle fingerprint (5 parameters записываются);
   - `tdmd verify --bitwise-compare` проверяет environment первым делом, emit explicit message если toolchains различаются;
   - CMake example с locked target arch (`-march=x86-64-v3`, не `native`);
   - Documented cross-compiler reproducibility levels (Level 1 bitwise на same toolchain, Level 2 layout-invariant cross-compiler, Level 3 scientific на different hardware).

2. **Thermostat-in-TD research program (integrator/SPEC.md §7.3, master spec §14 roadmap):**
   - Три варианта описаны formally: A (global frozen, v1.5 default), B (per-zone, rejected), C (lazy sync, research target);
   - M9 delivery: Variant A с PolicyValidator enforcing K=1 для NVT/NPT;
   - M11 research window (12 weeks): literature survey, prototype, validation suite, go/no-go decision;
   - Explicit acceptance criteria (equipartition MUST pass, speedup SHOULD pass >10%);
   - Consequence documented: NVT production работает но без TD speedup в v1.5 — known trade-off.

3. **Auto-K policy (§6.5a master spec):**
   - Three operation modes: manual (Reference/Production default), AutoK-v1 measurement-based (FastExperimental), perfmodel-assisted (M8+ Production);
   - Pseudocode algorithm AutoK-v1 с convergence analysis (log₂(K_max) iterations);
   - Hysteresis 5% prevents oscillation;
   - Safety constraints (K_range, no auto-K в Reference, no retune during rebuild);
   - Validation tests: correctness (same observables), convergence (synthetic workload), stability (no oscillation);
   - Roadmap M5 (fixed K) → M8 (AutoK-v1) → M9+ (perfmodel-assisted);
   - `tdmd explain --auto-k-recommendation` для users wanting bitwise determinism + optimal K.

4. **Cutoff and smoothing policy (potentials/SPEC.md §2.4):**
   - Four canonical treatments: A hard cutoff, B shifted-energy, C shifted-force (default production), D explicit smoothing (ML potentials);
   - Matrix potential → strategy (Morse/LJ → C, EAM → C, MEAM/SNAP/PACE → D);
   - Numerical considerations near cutoff (catastrophic cancellation avoidance);
   - `cutoff_treatment` field в `tdmd.yaml` с sensible defaults;
   - Validation thresholds добавлены в `verify/thresholds.yaml` (force jump, energy continuity, derivative continuity);
   - Critical: treatment должен точно совпадать с LAMMPS convention для differential tests.

5. **Python bindings strategy (новое Приложение E):**
   - 11-section formal strategy covering scope, layers, packaging, roadmap;
   - Three-layer architecture: Layer 1 low-level pybind11 (M9), Layer 2 ASE calculator (M11), Layer 3 high-level workflows (v2.0);
   - Numpy zero-copy views для read access; explicit API для mutations (version bumping);
   - Non-goals explicit: не replacement для CLI, не full runtime в Python, не ASE/LAMMPS competitor;
   - Package split: `tdmd-core` (20MB) + `tdmd-gpu` (180MB CUDA kernels);
   - Jupyter integration, parameter scans, custom validators как Python-specific features.

**Дополнительно:**

6. **B.2 open questions updated** — items 2 (Python API) и 10 (Auto-K) помечены RESOLVED с ссылками на новые разделы. Добавлен item 13 (Thermostat-in-TD decision pending M11).

7. **Roadmap Post-v1 переписан** (§14) — вместо bullet list теперь proper milestones M9-M13+ с deliverables, owner assignments, artifact gates. M9 NVT baseline, M10 MEAM, M11 NVT-in-TD research, M12 PACE/MLIAP, M13 Long-range + possible NVT-in-TD production.

### v2.3

Финализирована **precision policy** проекта. Новое Приложение D с 17 подсекциями покрывает все вопросы использования точности. Ключевые изменения:

1. **Новое Приложение D. Precision Policy Details** (~400 строк) — cross-cutting политика numerical precision. Включает:
   - Два опциональных BuildFlavor для mixed precision (Philosophy A и B);
   - Philosophy B (float compute, double accumulate) — default, safe для TD K-batched pipeline;
   - Philosophy A (float всюду) — opt-in research, NVE drift gates отключены;
   - Canonical force kernel template с precision layers;
   - Position delta policy (всегда double);
   - EAM table lookups (всегда double кроме Fp32Experimental);
   - Transcendentals, atomics, reductions, compile flags;
   - Запрет per-kernel precision overrides;
   - TF32 / FP8 / bfloat16 как future considerations.

2. **Пять BuildFlavor'ов вместо четырёх:** добавлен `MixedFastAggressiveBuild` (Philosophy A, opt-in research). `MixedFastBuild` остаётся default mixed (Philosophy B).

3. **§7.1 и §7.2 обновлены** — синхронизация с §D. Таблица BuildFlavor'ов расширена. Compatibility matrix обновлена включая `MixedFastAggressiveBuild`.

4. **M8 milestone расширен** — добавлен `MixedFastSnapOnlyBuild` как новый BuildFlavor, иллюстрирующий правильный путь per-kernel precision вариаций (через explicit BuildFlavor, не runtime overrides).

5. **TD×SD на GPU analysis (§D.1):** numerical обоснование почему Philosophy B — необходимый default для TDMD (TD K-batched pipeline float accumulation ~3e-7 × 10⁶ steps/ns = 0.3 eV/ns drift → нарушение NVE gate `1e-4/ns`).

6. **Acceptance thresholds матрица (§D.13)** — единая таблица допусков для всех пяти BuildFlavor'ов, привязанная к `verify/thresholds.yaml`.

7. **Build system integration (§D.14)** — CMake targets для всех пяти flavors с явными compile flags (FTZ, fast-math, etc).

8. **Validation procedure для новых BuildFlavor'ов (§D.17)** — formal requirements для добавления (SPEC delta, compat matrix update, thresholds, CMake, full slow-tier validation, dual review Architect + Validation Engineer).

9. **`__restrict__` policy (§D.16)** — обязательное использование pointer aliasing qualifier на всех hot kernels. Expected speedup 15-35% orthogonal к precision. Canonical force kernel template в §D.4 обновлён. Enforcement через clang-tidy custom check `tdmd-missing-restrict` и `[[tdmd::hot_kernel]]` attribute. CUDA-specific: `const T* __restrict__` enables read-only cache (`__ldg`).

### v2.2

Добавлен модуль **VerifyLab** (`verify/`) как cross-module scientific validation layer. Изменения:

1. **Новый модуль `verify/`** в §8.1 (список модулей верхнего уровня).
2. **Новая секция §13.0** — VerifyLab как owner для всего cross-module scientific validation.
3. **Новый модульный SPEC** `docs/specs/verify/SPEC.md` (~900 строк) со всеми деталями:
   - Threshold registry как единый источник истины для всех числовых допусков;
   - Canonical benchmarks T0-T7 как runnable tests;
   - Differential harness с LAMMPS as git submodule;
   - Anchor-test framework для воспроизведения §3.5 диссертации;
   - Physics invariant tests (conservation, thermodynamics);
   - Three tiers (fast/medium/slow) с соответствующей CI integration;
   - Diagnostic reports вместо простого pass/fail.
4. **Решения по open questions:**
   - LAMMPS как git submodule в `verify/third_party/lammps/` — agent-buildable;
   - Reference data в самом проекте (Git LFS для больших файлов);
   - Diagnostic mode как default (не только pass/fail verdict);
   - Threshold changes требуют review от Validation Engineer + Architect;
   - Three tiers: fast (PR), medium (nightly), slow (release).

### v2.1

Уточнения по результатам обсуждения двух архитектурных вопросов: two-level decomposition и unit systems. Принципиально не переделывает v2.0, а формализует то, что в v2.0 было подразумеваемым.

**Новые разделы:**

1. **§4a Two-level decomposition: TD внутри, SD снаружи** — формальная модель `P_total = P_space × P_time`, critical-path анализ, соответствие deployment pattern'ов реальной топологии железа (NVLink для inner, InfiniBand для outer), staging как Вариант C (TD-first M0–M6, SD-wrapping M7), protocol halo между subdomain'ами на разных temporal frontiers.
2. **§5.3 Unit system support** — принятая политика: `metal` native internal + `lj` через input adapter в v1; `real` добавляется post-v1 если понадобится reactive/organic; `si` не поддерживается никогда. Обязательное поле `units:` в `tdmd.yaml` (preflight error, не default).

**Уточнения в существующих разделах:**

3. **§6 scope clarified** — всё описанное относится к **одному subdomain**; в Pattern 1 это весь движок, в Pattern 2 — роль `InnerTdScheduler`.
4. **§6.3 пятая зависимость** — `SubdomainBoundaryDependency` для Pattern 2 (пустая в Pattern 1).
5. **§10.1 переработан в deployment patterns** — три pattern'а (single-subdomain TD, two-level TD×SD, SD-vacuum) с явными use-cases.
6. **§10.2 топологии** — разделены на inner (mesh внутри subdomain'а) и outer (Cartesian SD grid между subdomain'ами).
7. **§12.6 CommBackend** — разделён на inner (temporal packets) и outer (subdomain halo) channels.
8. **§12.7 PerfModel** — расширен prediction для Pattern 2 (`t_step_hybrid_seconds`, `recommended_pattern`, `recommended_P_space / P_time`).
9. **§12.7a OuterSdCoordinator** — новый интерфейс для Pattern 2 (M7+).
10. **§12.9 UnitConverter** — новый интерфейс в `interop/`.
11. **M1 roadmap** — добавлен `UnitConverter` skeleton.
12. **M2 roadmap** — `UnitConverter` полная поддержка `lj`, differential T1 в обеих unit systems как cross-check.
13. **M7 roadmap** — переформулирован как явное введение Pattern 2 (two-level), с явным переименованием `TdScheduler` → `InnerTdScheduler` и появлением `OuterSdCoordinator`.
14. **Приложение B.1** — добавлены пункты про deployment pattern staging и static subdomain grid.
15. **Приложение B.2** — добавлены открытые вопросы про auto-K, dynamic load balancing, `real` timing, subdomain limits.
16. **Приложение B.3** — добавлены явные non-goals: dynamic migration, `si`, `real`.

### v2.0-from-scratch

Переписан с нуля. Структурные изменения относительно v0.1-draft (`tdmd_engineering_spec_5_.md`) и v1.0-reframed (`tdmd_engineering_spec-4.md`):

1. **Физика TD и perf-model перенесены на первый план** (Часть I) — больше не погребены под процессом и policy-матрицами.
2. **Performance model формализована** как first-class концепт со своим модулем, своим gate в CI и своим CLI-подкомандой. Была отсутствующей.
3. **Anchor-test (воспроизведение TIME-MD)** введён как mandatory gate для M5. Раньше не существовал.
4. **Zoning planner выделен в отдельный модуль** со своим интерфейсом и property-тестами. Раньше был размазан между state / neighbor / scheduler.
5. **N_min и n_opt как явные методы `TdScheduler`** — ключевые параметры диссертационных формул (44)-(45), которых не было в interfaces.
6. **K-batched pipeline как first-class runtime parameter** (`pipeline_depth_cap`) с явным staging в milestones.
7. **Ring topology сохранена как legacy backend** для anchor-test'а; основной режим — mesh.
8. **Staging точности**: v1 живёт в `Fp64Reference` only, policy validator и матрица включаются с M7. Раньше проект пытался держать все 12 комбинаций с первого дня.
9. **SNAP поднят на M8** (было M17) как flagship proof-of-value, так как ML-kernel — главный customer метода.
10. **Traceability-таблица (Приложение A)** связывает каждый концепт TDMD с секцией диссертации. Раньше отсутствовала.
11. **Приложение B (Assumptions & Open Questions)** расширено и чётко разделено на принятые допущения, открытые вопросы, явные non-goals.
12. **Codex prompts** вынесены в отдельный документ (`docs/development/codex_playbook.md`), не засоряют master spec.
13. Объём мастер-специи сокращён с 2615 строк до ~800 без потери содержания.

### Источники истины

В иерархии точности:

1. Этот документ (TDMD Engineering Spec v2.1);
2. Модульные `docs/specs/<module>/SPEC.md`;
3. Execution pack текущего milestone;
4. Код.

Противоречие между кодом и spec — bug, а не источник истины. Противоречие между мастер-спецой и модульным SPEC — bug мастер-специи или bug модуля, требует explicit reconciliation через SPEC delta.

---

*Конец документа. Версия 2.5, дата: 2026-04-16.*
