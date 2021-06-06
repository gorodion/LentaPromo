# Описание задачи
Предоставлен перечень BigMedia промо-акций (билборды, фасады, обложки регулярных и сезонных каталогов) за последний квартал 2019-ого года и весь 2020-ый год, а также чековые данные для случайной выборки из клиентов "Ленты" за период с 20 сентября 2019 года по 20 сентября 2020 года и таблица с товарной иерархией.

**На основе этих данных, вам необходимо решить обе задачи:**

**1) Построить модель UpLift’а промо-кампаний и сделать предсказания для кампаний, проведённых в 4-ом квартале 2020-ого года**
UpLift для промо-кампании считается следующим образом: 
UpLift = (Total продажи в рублях в промопериоде / Total продажи в предпериоде) - 1, где предпериодом называется период, равный по продолжительности промопериоду, идущий непосредственно перед ним. Метрика для оценки качества - [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) (средняя абсолютная ошибка).

**2) Предложить оптимальный промо-календарь на первый квартал 2021-ого года**
Промо-календарь это набор промо-акций с указанием их типа, периода проведения, наполнения по SKU. Здесь ваша цель - максимизация общей выручки "Ленты" в первом квартале 2021-ого года (не только выручки от промо-кампаний). Решение второй задачи оценивается только качественно.

--------
UpLift - самая главное, предсказывать целиком по промо (а не по товарам). При этом учитывая состав товаров в промо и, в частности, насколько продаваемы товары попавшие в промо.

Календарь - основная идея в том, чтобы по каждому возможному периоду промо для товара предсказать добавку к прибыли компании которая вычисляется так:

Календарь реализовали с такими параметрами (логика представлена в презентации):
Только промо типа Biweekly, ровно одно промо в квартал по каждому SKU

Технически разбивали все на недели (начиная от четверга), вычисляли прибыль в промо и среднюю прибыль за неделю вне промо. Эта разница - добавка к прибыли компании при проведении промо. Для более точного моделирования нормировали таргет на средние продажи вне промо по данном SKU
