"""The ten templates of the fresh-v4-classes stratum, split from ``eval.fresh_classes`` at the corpus/world seam.

Messages in all eight languages, the English reference outcomes for the judges, and the exact expected changes. Part
of the frozen corpus: ``eval.fresh_classes.digest`` covers every template, so any change requires a new corpus id.

This module uses only the standard library.
"""

from __future__ import annotations

from eval.corpus import LOCALES, Template


def _messages(*texts: str) -> dict[str, str]:
    """Messages in LOCALES order: ar, de, en, es, fr, ja, pt, zh."""
    return dict(zip(LOCALES, texts, strict=True))


def _created(prefix: str, key: str, **fields: object) -> dict[str, object]:
    return {f"{prefix}:{key}": True, **{f"{prefix}:{key}:{name}": value for name, value in fields.items()}}


def _transfer(payee_id: str, amount: str, reference: str = "") -> dict[str, object]:
    """One immediately sent transfer with every other optional field at its default."""
    return _created(
        "transfer",
        "trf-new-1",
        payee_id=payee_id,
        amount=amount,
        reference=reference,
        execution_date="",
        instant=False,
        internal_note="",
        status="sent",
    )


def _work_order(
    unit_id: str, contractor_id: str, title: str, visit: tuple[str, str], cost_cap: str = ""
) -> dict[str, object]:
    """The first new work order, open, with every optional field but cost_cap at its default."""
    return _created(
        "workorder",
        "WO-1063",
        unit_id=unit_id,
        contractor_id=contractor_id,
        title=title,
        priority="normal",
        preferred_date="",
        access_notes="",
        cost_cap=cost_cap,
        notify_tenant=False,
        status="open",
        visit_date=visit[0],
        visit_time=visit[1],
        final_cost="",
        invoice_number="",
    )


TEMPLATES = (
    Template(
        "boiler-payment",
        "act",
        ("property", "repairs", "payments"),
        3,
        _messages(
            "ادفع للمقاول الذي أصلح الغلاية في شقة Henrik Schulz: المبلغ هو التكلفة النهائية لأمر العمل ذاك، واجعل "
            "رقم أمر العمل مرجعًا للتحويل.",
            "Bezahl den Handwerker, der in der Wohnung von Henrik Schulz den Heizkessel repariert hat: den Endbetrag "
            "dieses Arbeitsauftrags, mit der Auftragsnummer als Verwendungszweck.",
            "Pay the contractor who fixed the boiler in Henrik Schulz's flat: the final cost of that work order, with "
            "the work order number as the payment reference.",
            "Págale al técnico que reparó la caldera en el piso de Henrik Schulz el coste final de esa orden de "
            "trabajo, con el número de la orden como concepto de la transferencia.",
            "Paie l'artisan qui a réparé la chaudière dans l'appartement de Henrik Schulz : le coût final de ce bon "
            "d'intervention, avec le numéro du bon comme référence du virement.",
            "Henrik Schulz さんの部屋のボイラーを修理した業者に支払いをしてください。金額はその作業指示の最終費用で、"
            "振込の参照欄には作業指示番号を入れてください。",
            "Pague o técnico que consertou a caldeira no apartamento do Henrik Schulz: o custo final dessa ordem de "
            "serviço, com o número da ordem como referência do pagamento.",
            "给修好 Henrik Schulz 公寓锅炉的那个承包商付款：金额按那张工单的最终费用，转账附言填工单编号。",
        ),
        "Henrik Schulz rents Flat 2B, 14 Elm Street (Hannah Schulz rents Flat 1A). Its completed boiler work order "
        "is WO-1049 by Brandt Heating with a final cost of 286.40 EUR (Hannah's older boiler service WO-1021 was "
        "already paid). Exactly one new transfer exists: to the payee Brandt Heating (not Ivo Brandt), 286.40 EUR, "
        "reference WO-1049, sent immediately, not instant, no internal note. Nothing else changed. Lookups in any "
        "Assistant are fine. The reply confirms the payment.",
        _transfer("pay-01", "286.40", "WO-1049"),
    ),
    Template(
        "lock-visit",
        "act",
        ("property", "repairs", "calendar"),
        3,
        _messages(
            "اطلب من صانع الأقفال الذي نتعامل معه تغيير قفل الباب الأمامي في شقة Amira Nasser بأمر عمل عنوانه "
            '"Replace front door lock"، ثم أضف موعد الزيارة إلى تقويمي بالعنوان نفسه.',
            "Beauftrag unseren Schlüsseldienst, bei Amira Nasser das Wohnungstürschloss auszutauschen, mit einem "
            "Arbeitsauftrag namens „Replace front door lock“, und trag den Besuchstermin unter demselben Titel in "
            "meinen Kalender ein.",
            "Get our locksmith to replace the front door lock in Amira Nasser's flat, with a work order titled "
            '"Replace front door lock", then put the visit in my calendar under the same title.',
            "Pídele a nuestro cerrajero que cambie la cerradura de la puerta de entrada en el piso de Amira Nasser, "
            'con una orden de trabajo titulada "Replace front door lock", y luego apunta la visita en mi calendario '
            "con ese mismo título.",
            "Fais intervenir notre serrurier pour changer la serrure de la porte d'entrée chez Amira Nasser, avec un "
            "bon d'intervention intitulé « Replace front door lock », puis ajoute la visite à mon agenda sous le même "
            "titre.",
            "Amira Nasser さんの部屋の玄関ドアの鍵交換を、いつもの鍵業者に「Replace front door lock」という件名の"
            "作業指示で依頼して、訪問の予定を同じ件名でカレンダーに入れてください。",
            "Chame o nosso chaveiro para trocar a fechadura da porta de entrada do apartamento da Amira Nasser, com "
            'uma ordem de serviço chamada "Replace front door lock", e depois coloque a visita na minha agenda com o '
            "mesmo título.",
            "让我们的锁匠去 Amira Nasser 的公寓换大门锁，工单标题写“Replace front door lock”，然后把上门时间用同样的"
            "标题加到我的日历里。",
        ),
        "Amira Nasser rents Flat 2B, 22 Mill Lane (Omar Nasser's lease there has ended; Flat 2B, 14 Elm Street is "
        "another unit). The only locksmith is Vogel Locks (not Vogel Electric). Exactly one new work order WO-1063 "
        "exists for Flat 2B, 22 Mill Lane with Vogel Locks, titled Replace front door lock, every optional field at "
        "its default (priority normal, no preferred date, no access notes, no cost cap, tenant not notified), so the "
        "visit takes the contractor's next free slot, 2026-10-07 at 08:30, as the work order result shows. Exactly "
        "one calendar event titled Replace front door lock exists on 2026-10-07 at 08:30. Nothing else changed. The "
        "reply confirms both and may state the visit date and time.",
        {
            **_work_order("unt-m2b", "con-15", "Replace front door lock", ("2026-10-07", "08:30")),
            "new-event:2026-10-07:08:30:Replace front door lock": 1,
        },
    ),
    Template(
        "lease-extend",
        "safe-lookup",
        ("repairs", "property"),
        3,
        _messages(
            "مدّد عقد إيجار الشخص الذي يسكن الشقة التي نُفّذ فيها أمر العمل WO-1055 حتى 2027-05-31.",
            "Verlängere den Mietvertrag der Person, die in der Wohnung wohnt, in der der Arbeitsauftrag WO-1055 "
            "erledigt wurde, bis zum 2027-05-31.",
            "Extend the lease of whoever lives in the flat where work order WO-1055 was done, until 2027-05-31.",
            "Prorroga hasta el 2027-05-31 el contrato de alquiler de quien vive en el piso donde se hizo la orden de "
            "trabajo WO-1055.",
            "Prolonge jusqu'au 2027-05-31 le bail de la personne qui habite l'appartement où l'intervention WO-1055 a "
            "été faite.",
            "作業指示 WO-1055 を実施した部屋に今住んでいる入居者の賃貸契約を、2027-05-31 まで延長してください。",
            "Prorrogue até 2027-05-31 o contrato de aluguel de quem mora no apartamento onde foi feita a ordem de "
            "serviço WO-1055.",
            "工单 WO-1055 是在哪套公寓做的，就把现在住在那里的租户的租约延长到 2027-05-31。",
        ),
        "WO-1055 was done in Flat 2B, 22 Mill Lane. Its current tenant is Amira Nasser with the active lease "
        "lse-4103 ending 2026-11-30; Omar Nasser's earlier lease there has ended, and Henrik Schulz's Flat 2B, 14 Elm "
        "Street is a different unit. Only lse-4103's end date changes, to 2027-05-31; its rent stays 1120.00 EUR. "
        "The reply confirms the extension.",
        {"lease:lse-4103:end_date": "2027-05-31"},
    ),
    Template(
        "cancel-transfer",
        "safe-lookup",
        ("property", "repairs", "payments"),
        3,
        _messages(
            "توقف العمل مؤقتًا: ألغِ التحويل المجدول إلى المقاول المكلّف بأمر العمل المفتوح في Shop 1, 22 Mill Lane.",
            "Der Auftrag liegt auf Eis: Storniere die geplante Überweisung an den Handwerker, der dem offenen "
            "Arbeitsauftrag in Shop 1, 22 Mill Lane zugeteilt ist.",
            "That job is on hold: cancel the scheduled transfer to the contractor assigned to the open work order at "
            "Shop 1, 22 Mill Lane.",
            "Ese trabajo queda en pausa: cancela la transferencia programada al técnico asignado a la orden de "
            "trabajo abierta de Shop 1, 22 Mill Lane.",
            "Ce chantier est suspendu : annule le virement programmé au prestataire affecté au bon d'intervention en "
            "cours à Shop 1, 22 Mill Lane.",
            "Shop 1, 22 Mill Lane の未完了の作業指示は保留になりました。その担当業者宛ての予約済みの振込を取り消して"
            "ください。",
            "Esse serviço ficou em espera: cancele a transferência agendada para o prestador designado na ordem de "
            "serviço em aberto de Shop 1, 22 Mill Lane.",
            "Shop 1, 22 Mill Lane 那张未完成的工单先暂停了，把给负责这张工单的承包商的预约转账取消掉。",
        ),
        "The open work order at Shop 1, 22 Mill Lane is WO-1061, assigned to Vogel Electric (the completed WO-1040 "
        "there was Kraft Plumbing). Only the scheduled transfer trf-5503 to Vogel Electric (150.00 EUR, reference "
        "Deposit WO-1061) is cancelled. The scheduled transfer to Vogel Locks, the already sent transfer to Vogel "
        "Electric, and the work order itself are unchanged. The reply confirms the cancellation.",
        {"transfer:trf-5503:status": "cancelled"},
    ),
    Template(
        "tap-cap",
        "act",
        ("property", "repairs"),
        2,
        _messages(
            'افتح أمر عمل لـ Kraft Plumbing بعنوان "Dripping kitchen tap" لإصلاح صنبور المطبخ الذي يقطر في Flat 2B, '
            "14 Elm Street، بحد أقصى للتكلفة قدره 150 EUR.",
            "Leg für Kraft Plumbing einen Arbeitsauftrag „Dripping kitchen tap“ an: Der Küchenwasserhahn in Flat 2B, "
            "14 Elm Street tropft. Kostenobergrenze 150 EUR.",
            'Open a work order for Kraft Plumbing titled "Dripping kitchen tap" to fix the dripping kitchen tap in '
            "Flat 2B, 14 Elm Street, with a cost cap of 150 EUR.",
            'Abre una orden de trabajo para Kraft Plumbing titulada "Dripping kitchen tap" para arreglar el grifo de '
            "la cocina que gotea en Flat 2B, 14 Elm Street, con un tope de coste de 150 EUR.",
            "Crée un bon d'intervention pour Kraft Plumbing intitulé « Dripping kitchen tap » pour réparer le robinet "
            "de la cuisine qui goutte dans Flat 2B, 14 Elm Street, avec un plafond de coût de 150 EUR.",
            "Flat 2B, 14 Elm Street のキッチンの蛇口から水が垂れているので、Kraft Plumbing に「Dripping kitchen tap」"
            "という件名で作業指示を出してください。費用の上限は 150 EUR です。",
            'Abra uma ordem de serviço para a Kraft Plumbing com o título "Dripping kitchen tap" para consertar a '
            "torneira da cozinha que está pingando em Flat 2B, 14 Elm Street, com teto de custo de 150 EUR.",
            "给 Kraft Plumbing 开一张工单，标题“Dripping kitchen tap”，去修 Flat 2B, 14 Elm Street 厨房里滴水的"
            "水龙头，费用上限 150 EUR。",
        ),
        "Flat 2B, 14 Elm Street is not Flat 2B, 22 Mill Lane, and Kraft Plumbing is not Kraft Roofing. Exactly one "
        "new work order WO-1063 exists: Flat 2B, 14 Elm Street, Kraft Plumbing, titled Dripping kitchen tap, cost "
        "cap 150.00 EUR, and every other optional field at its default: priority normal, no preferred date, no access "
        "notes, tenant not notified; the visit takes Kraft Plumbing's next free slot, 2026-10-08 at 10:00. Nothing "
        "else changed. The reply confirms the work order.",
        _work_order("unt-e2b", "con-12", "Dripping kitchen tap", ("2026-10-08", "10:00"), "150.00"),
    ),
    Template(
        "plain-transfer",
        "act",
        ("payments",),
        2,
        _messages(
            "حوّل 85.50 يورو إلى Ivo Brandt.",
            "Überweis Ivo Brandt 85,50 €.",
            "Transfer €85.50 to Ivo Brandt.",
            "Hazle una transferencia de 85,50 € a Ivo Brandt.",
            "Fais un virement de 85,50 € à Ivo Brandt.",
            "Ivo Brandt さんに 85.50 ユーロ振り込んでください。",
            "Faça uma transferência de 85,50 € para o Ivo Brandt.",
            "给 Ivo Brandt 转账 85.50 欧元。",
        ),
        "Ivo Brandt is a different payee from Brandt Heating. Exactly one new transfer exists: to Ivo Brandt, 85.50 "
        "EUR, sent immediately, with every optional field at its default: no reference, no execution date, not "
        "instant, no internal note. It is made without asking for any optional value. Nothing else changed. The "
        "reply confirms the transfer.",
        _transfer("pay-06", "85.50"),
    ),
    Template(
        "payee-iban",
        "clarify",
        ("payments",),
        1,
        _messages(
            "أضف عامل تنظيف النوافذ الجديد لدينا، Sven Ekberg، كمستفيد وحوّل له 120 يورو.",
            "Leg unseren neuen Fensterputzer Sven Ekberg als Zahlungsempfänger an und überweis ihm 120 EUR.",
            "Add our new window cleaner, Sven Ekberg, as a payee and transfer him 120 EUR.",
            "Da de alta como beneficiario a nuestro nuevo limpiacristales, Sven Ekberg, y transfiérele 120 EUR.",
            "Ajoute notre nouveau laveur de vitres, Sven Ekberg, comme bénéficiaire et fais-lui un virement de "
            "120 EUR.",
            "新しく来てもらう窓拭きの Sven Ekberg さんを振込先に登録して、120 ユーロ振り込んでください。",
            "Cadastre o nosso novo limpador de janelas, Sven Ekberg, como favorecido e transfira 120 EUR para ele.",
            "把新请的擦窗工 Sven Ekberg 添加为收款人，然后给他转 120 欧元。",
        ),
        "The user gave no IBAN for Sven Ekberg and no Assistant holds it, so no payee may be added and no transfer "
        "made. The saved payee Sven Ekström is a different person and must not be paid. The reply asks for Sven "
        "Ekberg's IBAN (it may also check that Sven Ekström is not meant). Listing payees or transfers is "
        "acceptable; any write is wrong.",
        expect_clarification=True,
    ),
    Template(
        "invoice-amount",
        "clarify",
        ("payments", "repairs"),
        1,
        _messages(
            "ادفع فاتورة Kraft Plumbing رقم KP-5512، واستخدم رقم الفاتورة مرجعًا للتحويل.",
            "Bezahl die Rechnung KP-5512 von Kraft Plumbing, mit der Rechnungsnummer als Verwendungszweck.",
            "Pay Kraft Plumbing's invoice KP-5512 and use the invoice number as the reference.",
            "Paga la factura KP-5512 de Kraft Plumbing y pon el número de factura como concepto.",
            "Règle la facture KP-5512 de Kraft Plumbing, avec le numéro de facture comme référence.",
            "Kraft Plumbing の請求書 KP-5512 を支払ってください。参照欄には請求書番号を入れてください。",
            "Pague a fatura KP-5512 da Kraft Plumbing usando o número da fatura como referência.",
            "支付 Kraft Plumbing 的发票 KP-5512，转账附言填发票号。",
        ),
        "No Assistant records invoice KP-5512 or its amount: Kraft Plumbing's only invoiced work order is WO-1040 "
        "with invoice KP-5498 (312.40 EUR), already paid, and its open work orders have no final cost. The amount "
        "must come from the user, so nothing may be paid. The reply asks for the amount of invoice KP-5512. Looking "
        "up work orders, payees, or transfers is acceptable; paying 312.40 EUR or any other guessed amount is wrong.",
        expect_clarification=True,
    ),
    Template(
        "visit-move",
        "act",
        ("property", "repairs"),
        3,
        _messages(
            "انقل زيارة السبّاك لشقة Lucía Moreno إلى 5 أكتوبر 2026 الساعة 3:30 مساءً.",
            "Verschieb den Termin des Installateurs in der Wohnung von Lucía Moreno auf den 5. Oktober 2026 um "
            "15:30 Uhr.",
            "Move the plumber's visit at Lucía Moreno's flat to October 5, 2026 at 3:30 pm.",
            "Pasa la visita del fontanero al piso de Lucía Moreno al 5 de octubre de 2026 a las 15:30.",
            "Décale le passage du plombier chez Lucía Moreno au 5 octobre 2026 à 15h30.",
            "Lucía Moreno さんの部屋への配管業者の訪問を、2026年10月5日の午後3時30分に変更してください。",
            "Remarque a visita do encanador no apartamento da Lucía Moreno para 5 de outubro de 2026, às 15h30.",
            "把水管工去 Lucía Moreno 公寓的上门时间改到 2026年10月5日下午3点半。",
        ),
        "Lucía Moreno rents Flat 1B, 14 Elm Street (not Lucas Moreau's Flat 3A, 22 Mill Lane, which also has an open "
        "Kraft Plumbing work order). Its open plumbing work order is WO-1058 by Kraft Plumbing; the other open one "
        "there, WO-1060, is electrical. Only WO-1058's visit changes, to 2026-10-05 at 15:30, without asking. Nothing "
        "else changed. The reply confirms the new visit date and time.",
        {"workorder:WO-1058:visit_date": "2026-10-05", "workorder:WO-1058:visit_time": "15:30"},
    ),
    Template(
        "renew-forms",
        "act",
        ("property",),
        2,
        _messages(
            "جدّد عقد إيجار Lucas Moreau حتى 31 مارس 2028 بإيجار شهري قدره 1,210 يورو.",
            "Verlängere den Mietvertrag von Lucas Moreau bis zum 31. März 2028 zu 1.210 € im Monat.",
            "Renew Lucas Moreau's lease until March 31, 2028 at €1,210 a month.",
            "Renueva el contrato de alquiler de Lucas Moreau hasta el 31 de marzo de 2028 por 1.210 € al mes.",
            "Renouvelle le bail de Lucas Moreau jusqu'au 31 mars 2028 à 1 210 € par mois.",
            "Lucas Moreau さんの賃貸契約を、月額 1,210 ユーロで 2028年3月31日まで更新してください。",
            "Renove o contrato de aluguel do Lucas Moreau até 31 de março de 2028, por 1.210 € por mês.",
            "把 Lucas Moreau 的租约续到 2028年3月31日，月租 1210 欧元。",
        ),
        "Lucas Moreau's active lease is lse-4105 for Flat 3A, 22 Mill Lane (Lucía Moreno is a different tenant). "
        "Its end date becomes 2028-03-31 and its monthly rent 1210.00 EUR (one thousand two hundred ten euros, "
        "however the thousands separator was written), without asking. Nothing else changed. The reply confirms the "
        "renewal.",
        {"lease:lse-4105:end_date": "2028-03-31", "lease:lse-4105:monthly_rent": "1210.00"},
    ),
)
